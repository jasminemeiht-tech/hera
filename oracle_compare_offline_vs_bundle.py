import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

# Paths
ORACLE_DIR = Path("/home/ubuntu/ATHENA/ORACLE").resolve()
JSONL_PATH = Path(
    "/home/ubuntu/ATHENA/DATASET/dataset_served_alpaca/"
    "batch-01_20251113_180031/train/fold_1.jsonl"
)
CKPT_PATH = Path(
    "/home/ubuntu/ATHENA/ORACLE/__pycache__/full_run_train_batch_3/"
    "len_pred_full.pth"
)
BEST_BUNDLE_DIR = Path("/home/ubuntu/ATHENA/ORACLE/oracle_model_best")
MAX_SAMPLES = 500

if str(ORACLE_DIR) not in sys.path:
    sys.path.append(str(ORACLE_DIR))

from len_pred import (  # type: ignore  # noqa: E402
    EnhancedV7EightClassDataset,
    EnhancedV7ModelRebuilder,
    compute_enhanced_v7_classification_metrics,
)

from vllm.oracle_lenpred import (  # noqa: E402
    predict_length as oracle_predict_length,
)


def _set_oracle_env() -> None:
    os.environ.setdefault("ORACLE_LENPRED_SCRIPT",
                          str(ORACLE_DIR / "len_pred.py"))
    os.environ.setdefault("ORACLE_LENPRED_BUNDLE_DIR", str(BEST_BUNDLE_DIR))
    os.environ.setdefault("ORACLE_LENPRED_DEVICE", "cpu")
    os.environ.setdefault("ORACLE_LENPRED_ENABLED", "1")


def run_compare() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading checkpoint from: {CKPT_PATH}")

    # 1) 从 best checkpoint 重建训练侧系统
    system = EnhancedV7ModelRebuilder.rebuild_from_checkpoint(
        str(CKPT_PATH),
        device=str(device),
        verify_functionality=False,
        restore_training_state=False,
    )

    tokenizer = system.feature_extractor.tokenizer
    discretizer = system.discretizer

    # 2) 构造与训练完全一致的 Dataset + DataLoader
    dataset = EnhancedV7EightClassDataset(
        jsonl_files=[str(JSONL_PATH)],
        tokenizer=tokenizer,
        discretizer=discretizer,
        max_length=512,
        boundary_sample_weight=1.2,
        use_soft_labels=False,
    )
    data_collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=1,  # 方便逐条取 prompt
        shuffle=False,
        num_workers=0,
        collate_fn=data_collator,
    )

    print(f"[INFO] Dataset size: {len(dataset)}")

    # 3) 设置 oracle bundle 环境变量
    _set_oracle_env()

    # 收集对比结果
    true_lengths: List[float] = []
    pred_classes_train: List[int] = []
    pred_classes_bundle: List[int] = []
    reg_train: List[float] = []
    reg_bundle: List[float] = []

    # 训练侧：用 system.predict(prompt_text)
    system.feature_extractor.set_training_mode(False)
    system.predictor.eval()

    # 直接从 dataset.data 取 prompt / output_length，确保和训练一致
    # （DataLoader 只用来保证 tokenizer/padding 行为）
    for idx in range(min(MAX_SAMPLES, len(dataset))):
        item: Dict[str, Any] = dataset.data[idx]  # type: ignore[attr-defined]
        prompt_text: str = item["prompt"]
        output_len: float = float(item["output_length"])

        # 训练侧预测
        res_train = system.predict(prompt_text, return_distribution=False)
        cls_train = int(res_train["classification"]["predicted_class"])
        reg_train_len = float(res_train["regression"]["estimated_length"])

        # Bundle / vLLM oracle 预测（使用相同的 prompt_text）
        meta = oracle_predict_length(prompt_text)
        if meta is None:
            print(f"[WARN] oracle_predict_length returned None at idx={idx}")
            continue
        cls_bundle = meta.get("class_index")
        reg_bundle_len = meta.get("regression_estimated_length")
        if cls_bundle is None or reg_bundle_len is None:
            print(
                f"[WARN] oracle_metadata incomplete at idx={idx}, meta={meta}")
            continue

        true_lengths.append(output_len)
        pred_classes_train.append(cls_train)
        pred_classes_bundle.append(int(cls_bundle))
        reg_train.append(reg_train_len)
        reg_bundle.append(float(reg_bundle_len))

        if idx < 5:
            print(f"\n[Sample {idx}]")
            print(f"  true_len           = {output_len}")
            print(f"  train_cls, bundle_cls = {cls_train}, {cls_bundle}")
            print(f"  train_reg, bundle_reg = {reg_train_len:.2f}, "
                  f"{reg_bundle_len:.2f}")

    # 转成 numpy
    tl = np.array(true_lengths, dtype=float)
    cls_t = np.array(pred_classes_train, dtype=int)
    cls_b = np.array(pred_classes_bundle, dtype=int)
    reg_t = np.array(reg_train, dtype=float)
    reg_b = np.array(reg_bundle, dtype=float)

    print(f"\n[INFO] Collected {len(tl)} valid comparison samples.")

    # 4) 训练侧 vs 真值 的指标
    metrics_train = compute_enhanced_v7_classification_metrics(
        y_true_lengths=tl,
        y_pred_classes=cls_t,
        y_prob=np.zeros((len(tl), discretizer.num_bins), dtype=float),
        discretizer=discretizer,
        stage_name="TrainSide",
    )
    print("\n=== Train-side metrics vs true (fold_1 subset) ===")
    print(f"overall_accuracy (strict): {metrics_train['overall_accuracy']:.4f}")
    print(f"enhanced_accuracy        : {metrics_train['enhanced_accuracy']:.4f}")

    # 5) Bundle oracle vs 真值 的指标
    metrics_bundle = compute_enhanced_v7_classification_metrics(
        y_true_lengths=tl,
        y_pred_classes=cls_b,
        y_prob=np.zeros((len(tl), discretizer.num_bins), dtype=float),
        discretizer=discretizer,
        stage_name="BundleSide",
    )
    print("\n=== Bundle-oracle metrics vs true (fold_1 subset) ===")
    print(
        f"overall_accuracy (strict): {metrics_bundle['overall_accuracy']:.4f}")
    print(
        f"enhanced_accuracy        : {metrics_bundle['enhanced_accuracy']:.4f}")

    # 6) 训练侧预测 vs Bundle oracle 预测 一致性
    cls_agree = np.mean(cls_t == cls_b)
    reg_mae = float(np.mean(np.abs(reg_t - reg_b)))
    print("\n=== Agreement between train-side and bundle oracle ===")
    print(f"classification match rate : {cls_agree:.4f}")
    print(f"regression MAE            : {reg_mae:.4f}")


if __name__ == "__main__":
    run_compare()

