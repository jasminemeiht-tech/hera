# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import os
import threading
from typing import Any, Dict, Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ORACLE_LOCK = threading.Lock()
_ORACLE_INSTANCE: Optional[Any] = None
_ORACLE_FAILED: bool = False


def _is_enabled() -> bool:
    """Return whether oracle integration is enabled via env."""
    return os.getenv("ORACLE_LENPRED_ENABLED", "1") not in ("0", "false",
                                                           "False")


def _load_oracle() -> Optional[Any]:
    """Lazily load OracleServing from an external bundle.

    This uses export_for_serving_from_checkpoint() from the ORACLE project.
    To configure paths and device:
      - ORACLE_LENPRED_SCRIPT: path to len_pred.py (default:
        /home/ubuntu/ATHENA/ORACLE/len_pred.py)
      - ORACLE_LENPRED_BUNDLE_DIR: serving bundle dir (default:
        /home/ubuntu/ATHENA/ORACLE/oracle_model)
      - ORACLE_LENPRED_DEVICE: device string, e.g. "cuda" or "cpu"
    """
    global _ORACLE_INSTANCE, _ORACLE_FAILED

    if _ORACLE_INSTANCE is not None or _ORACLE_FAILED:
        return _ORACLE_INSTANCE

    if not _is_enabled():
        _ORACLE_FAILED = True
        return None

    script_path = os.getenv("ORACLE_LENPRED_SCRIPT",
                            "/home/ubuntu/ATHENA/ORACLE/len_pred.py")
    bundle_dir = os.getenv("ORACLE_LENPRED_BUNDLE_DIR",
                           "/home/ubuntu/ATHENA/ORACLE/oracle_model")

    if not os.path.isfile(script_path):
        logger.warning("Oracle lenpred script not found at %s; "
                       "disabling oracle integration.", script_path)
        _ORACLE_FAILED = True
        return None

    manifest_path = os.path.join(bundle_dir, "manifest", "config.json")
    if not os.path.isfile(manifest_path):
        logger.warning(
            "Oracle serving bundle manifest not found at %s; "
            "please export the bundle from the checkpoint and set "
            "ORACLE_LENPRED_BUNDLE_DIR accordingly. Disabling oracle "
            "integration for now.", manifest_path)
        _ORACLE_FAILED = True
        return None

    device_env = os.getenv("ORACLE_LENPRED_DEVICE")
    if device_env is not None:
        device = device_env
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with _ORACLE_LOCK:
        if _ORACLE_INSTANCE is not None or _ORACLE_FAILED:
            return _ORACLE_INSTANCE

        try:
            spec = importlib.util.spec_from_file_location(
                "oracle_len_pred", script_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load spec from {script_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[arg-type]

            OracleServing = getattr(module, "OracleServing", None)
            if OracleServing is None:
                raise ImportError(
                    "OracleServing class not found in len_pred.py")

            oracle = OracleServing.from_bundle(bundle_dir=bundle_dir,
                                               device=device)

            # Wrap the feature extractor to make it robust to
            # AutoModelForCausalLM outputs, mimicking the training-time
            # EnhancedV7FeatureExtractor.extract_features behavior.
            fe = getattr(oracle, "feature_extractor", None)

            if fe is not None and hasattr(fe, "model"):

                class _WrappedFeatureExtractor:

                    def __init__(self, inner: Any):
                        self._inner = inner
                        self.model = inner.model
                        self.tokenizer = inner.tokenizer
                        self.device = inner.device

                    def set_training_mode(self, training: bool) -> None:
                        # Delegate to original extractor
                        if hasattr(self._inner, "set_training_mode"):
                            self._inner.set_training_mode(training)
                        else:
                            self.model.train(training)

                    def extract_features(self,
                                         input_ids: torch.Tensor,
                                         attention_mask: torch.Tensor,
                                         training: bool = False
                                         ) -> torch.Tensor:
                        # For CausalLM models, prefer the internal base model
                        backbone = getattr(self.model, "model", self.model)
                        if training:
                            outputs = backbone(
                                input_ids=input_ids.to(self.device),
                                attention_mask=attention_mask.to(self.device),
                            )
                        else:
                            with torch.no_grad():
                                outputs = backbone(
                                    input_ids=input_ids.to(self.device),
                                    attention_mask=attention_mask.to(
                                        self.device),
                                )

                        # Handle BaseModelOutputWithPast / CausalLMOutputWithPast /
                        # tuple outputs.
                        if hasattr(outputs, "last_hidden_state"):
                            last_hidden_state = outputs.last_hidden_state
                        elif getattr(outputs, "hidden_states",
                                     None) is not None:
                            # Use the last layer hidden state if available.
                            last_hidden_state = outputs.hidden_states[-1]
                        else:
                            # Fall back to first element of tuple-like outputs.
                            last_hidden_state = outputs[0]

                        return last_hidden_state

                oracle.feature_extractor = _WrappedFeatureExtractor(fe)

            _ORACLE_INSTANCE = oracle
            logger.info(
                "Loaded oracle length predictor from %s (bundle=%s, device=%s)",
                script_path,
                bundle_dir,
                device,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Failed to initialize oracle length predictor; "
                "oracle integration will be disabled. Error: %s", exc)
            _ORACLE_FAILED = True
            _ORACLE_INSTANCE = None

        return _ORACLE_INSTANCE


def predict_length(prompt: str) -> Optional[Dict[str, Any]]:
    """Run oracle length prediction for a single prompt.

    Returns a compact, engine-agnostic metadata dict suitable
    for attaching to vLLM request objects, or None if prediction
    is unavailable or disabled.
    """
    if not _is_enabled():
        return None

    oracle = _load_oracle()
    if oracle is None:
        return None

    try:
        result = oracle.predict(prompt)  # type: ignore[call-arg]
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Oracle prediction failed; ignoring for this request. "
                       "Error: %s", exc)
        return None

    try:
        classification = result.get("classification", {}) if isinstance(
            result, dict) else {}
        regression = result.get("regression", {}) if isinstance(
            result, dict) else {}
        scheduling = result.get("scheduling", {}) if isinstance(
            result, dict) else {}
        prompt_info = result.get("prompt_info", {}) if isinstance(
            result, dict) else {}
        model_info = result.get("model_info", {}) if isinstance(
            result, dict) else {}

        meta: Dict[str, Any] = {
            "class_index": classification.get("predicted_class"),
            "class_name": classification.get("class_name"),
            "class_confidence": classification.get("confidence"),
            "class_estimated_length": classification.get("estimated_length"),
            "regression_estimated_length": regression.get("estimated_length"),
            "regression_mean": regression.get("predicted_mean"),
            "scheduling": scheduling,
            "prompt_length": prompt_info.get("prompt_length"),
            "oracle_model_info": model_info,
        }
        return meta
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "Failed to normalize oracle prediction output; "
            "raw result will be dropped. Error: %s", exc)
        return None
