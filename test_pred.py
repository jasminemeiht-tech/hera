import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
DATA_PATH = Path(
    "/home/ubuntu/ATHENA/DATASET/dataset_served_alpaca/"
    "batch-01_20251113_180031/train/fold_1.jsonl")
ORACLE_SCRIPT = Path("/home/ubuntu/ATHENA/ORACLE/len_pred.py")
ORACLE_BUNDLE = Path("/home/ubuntu/ATHENA/ORACLE/oracle_model")
LLAMA_MODEL_DIR = Path(
    "/home/ubuntu/ATHENA/MODEL_SET/meta-llama/Llama-3.1-8B-Instruct")

MAX_SAMPLES = 500

# Ensure oracle integration is configured (CPU for oracle by default).
os.environ.setdefault("ORACLE_LENPRED_SCRIPT", str(ORACLE_SCRIPT))
os.environ.setdefault("ORACLE_LENPRED_BUNDLE_DIR", str(ORACLE_BUNDLE))
os.environ.setdefault("ORACLE_LENPRED_DEVICE", "cpu")
os.environ.setdefault("ORACLE_LENPRED_ENABLED", "1")

from vllm.oracle_lenpred import predict_length as oracle_predict_length  # noqa: E402

# Eight-class discretizer bin edges (tokens), aligned with oracle training.
BIN_EDGES = np.array(
    [0, 40, 80, 120, 180, 260, 360, 520, float("inf")], dtype=float)

# Enhanced v7 tolerance rules（与 len_pred.py 保持一致）
TOLERANCE_RULES = {
    0: {"max_correct": 60},
    1: {"min_correct": 30, "max_correct": 90},
    2: {"min_correct": 60, "max_correct": 140},
    3: {"min_correct": 100, "max_correct": 220},
    4: {"min_correct": 160, "max_correct": 300},
    5: {"min_correct": 220, "max_correct": 380},
    6: {"min_correct": 320, "max_correct": 520},
    7: {"min_correct": 440},
}


def length_to_class(length: float) -> int:
    """Map a token length to discretizer class index."""
    idx = int(np.digitize([length], BIN_EDGES)[0]) - 1
    idx = max(0, min(idx, len(BIN_EDGES) - 2))
    return idx


def is_enhanced_correct(true_length: float, predicted_class: int) -> bool:
    """Enhanced v7 容错正确性判定（复刻 len_pred.py 的逻辑）。"""
    true_cls = length_to_class(true_length)
    if predicted_class == true_cls:
        return True

    rule = TOLERANCE_RULES.get(predicted_class)
    if not rule:
        return False

    min_correct = float(rule.get("min_correct", 0.0))
    max_correct = float(rule.get("max_correct", float("inf")))
    return (min_correct <= true_length <= max_correct)


def load_samples(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            samples.append(sample)
    return samples


def build_prompt_from_messages(sample: Dict[str, Any]) -> Optional[str]:
    """Build a simple prompt string from dataset messages.

    For这次评估，我们使用最后一个 user 消息的 content 作为 prompt，
    与 oracle 的训练风格保持一致，也方便与 Llama 生成对齐。
    """
    messages = sample.get("messages") or []
    user_contents: List[str] = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                user_contents.append(content.strip())
    if not user_contents:
        return None
    # 多轮时简单串联
    return "\n\n".join(user_contents)


def load_llama_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading Llama model from {LLAMA_MODEL_DIR} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(LLAMA_MODEL_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(LLAMA_MODEL_DIR),
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_length_llama(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> int:
    """Generate with Llama and return number of generated tokens."""
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0, input_ids.shape[1]:]
    return int(gen_ids.shape[0])


def compute_stats(
    name: str,
    true_lengths: List[int],
    pred_reg_lengths: List[float],
    pred_class_indices: List[int],
) -> None:
    """Compute error distribution and per-class recall."""
    assert len(true_lengths) == len(pred_reg_lengths) == len(
        pred_class_indices)

    n = len(true_lengths)
    if n == 0:
        print(f"[WARN] No samples for {name}")
        return

    errors = [pred_reg_lengths[i] - true_lengths[i] for i in range(n)]
    abs_errors = [abs(e) for e in errors]

    def quantile(vals: List[float], q: float) -> float:
        if not vals:
            return float("nan")
        return float(np.quantile(np.array(vals, dtype=float), q))

    mean_err = float(np.mean(errors))
    mae = float(np.mean(abs_errors))
    med_ae = quantile(abs_errors, 0.5)
    p90 = quantile(abs_errors, 0.9)
    p95 = quantile(abs_errors, 0.95)

    print(f"\n=== [{name}] 回归误差统计（pred_reg vs true） ===")
    print(f"样本数: {n}")
    print(f"平均误差 (pred - true): {mean_err:.3f}")
    print(f"平均绝对误差 MAE: {mae:.3f}")
    print(f"绝对误差中位数: {med_ae:.3f}")
    print(f"绝对误差 90 分位: {p90:.3f}")
    print(f"绝对误差 95 分位: {p95:.3f}")

    # 简单误差区间分布
    bins = [0, 10, 20, 40, 80, 160, float("inf")]
    bin_labels = ["<=10", "10-20", "20-40", "40-80", "80-160", ">160"]
    bin_counts = Counter()
    for ae in abs_errors:
        for i in range(len(bins) - 1):
            if bins[i] <= ae < bins[i + 1]:
                bin_counts[bin_labels[i]] += 1
                break
    print("\n误差分布 (按绝对误差区间):")
    for label in bin_labels:
        cnt = bin_counts.get(label, 0)
        print(f"  {label}: {cnt} ({cnt / n * 100:.1f}%)")

    # 分类档位召回率
    true_classes = [length_to_class(t) for t in true_lengths]
    total_correct = 0
    total_enhanced_correct = 0
    class_support = Counter()
    class_correct = Counter()
    class_enhanced_correct = Counter()
    for i, (tc, pc) in enumerate(zip(true_classes, pred_class_indices)):
        class_support[tc] += 1
        if tc == pc:
            class_correct[tc] += 1
            total_correct += 1
        if is_enhanced_correct(true_length=true_lengths[i],
                               predicted_class=pc):
            class_enhanced_correct[tc] += 1
            total_enhanced_correct += 1

    overall_acc = total_correct / n
    enhanced_acc = total_enhanced_correct / n
    print(f"\n[{name}] 分类整体准确率 (strict): {overall_acc:.4f}")
    print(f"[{name}] Enhanced accuracy: {enhanced_acc:.4f}")
    print("按档位的召回率（strict，基于真值长度所在档位）：")
    for cls_idx in range(len(BIN_EDGES) - 1):
        support = class_support.get(cls_idx, 0)
        correct = class_correct.get(cls_idx, 0)
        if support > 0:
            recall = correct / support
            print(
                f"  Class {cls_idx}: support={support}, correct={correct}, recall={recall:.4f}"
            )
        else:
            print(f"  Class {cls_idx}: support=0, correct=0, recall=N/A")

    print("\n按档位的 Enhanced recall：")
    for cls_idx in range(len(BIN_EDGES) - 1):
        support = class_support.get(cls_idx, 0)
        correct_e = class_enhanced_correct.get(cls_idx, 0)
        if support > 0:
            recall_e = correct_e / support
            print(
                f"  Class {cls_idx}: support={support}, enhanced_correct={correct_e}, enhanced_recall={recall_e:.4f}"
            )
        else:
            print(f"  Class {cls_idx}: support=0, enhanced_correct=0, enhanced_recall=N/A")


def main() -> None:
    print(f"[INFO] Loading samples from {DATA_PATH} (first {MAX_SAMPLES})...")
    samples = load_samples(DATA_PATH, MAX_SAMPLES)
    print(f"[INFO] Loaded {len(samples)} samples.")

    print("[INFO] Loading Llama-3.1-8B-Instruct...")
    llama_tokenizer, llama_model, llama_device = load_llama_model()

    # Containers for two evaluation modes
    true_lengths_token_counts: List[int] = []
    true_lengths_llama: List[int] = []
    pred_reg_all: List[float] = []
    pred_class_all: List[int] = []

    pred_reg_llama: List[float] = []
    pred_class_llama: List[int] = []

    # For sanity printing
    preview_rows: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        prompt = build_prompt_from_messages(sample)
        if not prompt:
            continue

        # Oracle prediction
        oracle_meta = oracle_predict_length(prompt)
        if oracle_meta is None:
            print(
                f"[WARN] Oracle prediction failed on sample {idx}, skipping.")
            continue

        pred_class = oracle_meta.get("class_index")
        pred_reg_len = oracle_meta.get("regression_estimated_length")
        if pred_class is None or pred_reg_len is None:
            print(
                f"[WARN] Oracle metadata incomplete on sample {idx}, skipping."
            )
            continue

        # Ground truth 1: dataset token_counts["output"]
        token_counts = sample.get("token_counts") or {}
        true_len_token = int(token_counts.get("output", 0))

        # Ground truth 2: Llama-3.1-8B-Instruct actual generation length
        decode_cfg = sample.get("decode_config") or {}
        # 按要求固定采样参数（忽略文件中可能的差异）
        max_new_tokens = 512
        temperature = 0.2
        top_p = 0.9

        gen_len_llama = generate_length_llama(
            llama_tokenizer,
            llama_model,
            llama_device,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        true_lengths_token_counts.append(true_len_token)
        pred_reg_all.append(float(pred_reg_len))
        pred_class_all.append(int(pred_class))

        true_lengths_llama.append(gen_len_llama)
        pred_reg_llama.append(float(pred_reg_len))
        pred_class_llama.append(int(pred_class))

        if len(preview_rows) < 5:
            preview_rows.append({
                "sample_index": idx,
                "prompt_preview": prompt[:80].replace("\n", " "),
                "true_len_token_counts": true_len_token,
                "true_len_llama": gen_len_llama,
                "pred_class": int(pred_class),
                "pred_reg_len": float(pred_reg_len),
            })

    # Sanity check output
    print("\n=== 样例输出预览（前 5 条） ===")
    for row in preview_rows:
        print(json.dumps(row, ensure_ascii=False))

    # Evaluation 1: use dataset token_counts["output"] as ground truth
    compute_stats(
        name="真值=token_counts.output",
        true_lengths=true_lengths_token_counts,
        pred_reg_lengths=pred_reg_all,
        pred_class_indices=pred_class_all,
    )

    # Evaluation 2: use Llama-3.1-8B-Instruct actual generation length
    compute_stats(
        name="真值=Llama-3.1-8B-Instruct 生成长度",
        true_lengths=true_lengths_llama,
        pred_reg_lengths=pred_reg_llama,
        pred_class_indices=pred_class_llama,
    )


if __name__ == "__main__":
    main()
