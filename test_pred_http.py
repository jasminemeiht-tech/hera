import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

# 让我们可以复用生成 Alpaca 数据时的 SYSTEM_PROMPT 定义
ALPACA_SERVE_DIR = Path(
    "/home/ubuntu/ATHENA/DATASET/alpaca_serve_code").resolve()
if str(ALPACA_SERVE_DIR) not in sys.path:
    sys.path.append(str(ALPACA_SERVE_DIR))

try:
    from serve_alpaca import SYSTEM_PROMPT
except Exception:  # noqa: BLE001
    SYSTEM_PROMPT = (
        "You are a helpful assistant used to generate training data for a "
        "length-aware system."
    )

DATA_PATH = Path(
    "/home/ubuntu/ATHENA/DATASET/dataset_served_alpaca/"
    # "batch-01_20251113_180031/train/fold_1.jsonl"
    "batch-03_20251113_200822/train/fold_1.jsonl")

API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instruct"  # 对应你启动 server 时的 --served-model-name

MAX_SAMPLES = 500

# 与 oracle 训练保持一致的分箱
BIN_EDGES = np.array(
    [0, 40, 80, 120, 180, 260, 360, 520, float("inf")], dtype=float)

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
    idx = int(np.digitize([length], BIN_EDGES)[0]) - 1
    idx = max(0, min(idx, len(BIN_EDGES) - 2))
    return idx


def is_enhanced_correct(true_length: float, predicted_class: int) -> bool:
    true_cls = length_to_class(true_length)
    if predicted_class == true_cls:
        return True
    rule = TOLERANCE_RULES.get(predicted_class)
    if not rule:
        return False
    min_correct = float(rule.get("min_correct", 0.0))
    max_correct = float(rule.get("max_correct", float("inf")))
    return min_correct <= true_length <= max_correct


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


def build_chat_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """构造与 serve_alpaca.py 相同风格的对话消息。

    训练 oracle 时，Alpaca 样本通过 build_messages(rec) 变为：
      - system: SYSTEM_PROMPT（较长的指导语）
      - user: instruction + 可选的 Input: 段

    在 served 数据集中，我们保存的是 sanitized messages：
      - system: 仅保留 role，不含 content
      - user: content 已经是上述 user_content（instruction [+ Input...]）

    这里我们复原当时的对话形态：重新填回 SYSTEM_PROMPT 作为 system content，
    user content 直接复用样本中的 user 消息 content。
    """
    msgs = sample.get("messages") or []
    user_content = ""
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            user_content = (m.get("content") or "").strip()
            break

    if not user_content:
        # 兜底：没有 user 消息时，使用 output 作为 prompt
        user_content = (sample.get("output") or "").strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


async def call_openai_chat(
    client: httpx.AsyncClient,
    messages: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 512,
        "stream": False,
    }
    try:
        resp = await client.post(API_URL, json=payload)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] HTTP request failed: {exc}")
        return None

    if resp.status_code != 200:
        print(f"[WARN] HTTP {resp.status_code}: {resp.text[:200]}...")
        return None
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to parse JSON response: {exc}")
        return None


def compute_stats(
    name: str,
    true_lengths: List[float],
    pred_reg_lengths: List[float],
    pred_class_indices: List[int],
) -> None:
    n = len(true_lengths)
    if n == 0:
        print(f"\n[{name}] 无有效样本，跳过统计。")
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

    bins = [0, 10, 20, 40, 80, 160, float("inf")]
    labels = ["<=10", "10-20", "20-40", "40-80", "80-160", ">160"]
    bin_counts = Counter()
    for ae in abs_errors:
        for i in range(len(bins) - 1):
            if bins[i] <= ae < bins[i + 1]:
                bin_counts[labels[i]] += 1
                break
    print("\n误差分布 (按绝对误差区间):")
    for label in labels:
        cnt = bin_counts.get(label, 0)
        print(f"  {label}: {cnt} ({cnt / n * 100:.1f}%)")

    # 分类准确率和 per-class recall
    true_classes = [length_to_class(t) for t in true_lengths]
    total_correct = 0
    total_enhanced_correct = 0
    support = Counter()
    correct = Counter()
    enhanced_correct = Counter()

    for i, (tc, pc) in enumerate(zip(true_classes, pred_class_indices)):
        tlen = true_lengths[i]
        support[tc] += 1
        if pc == tc:
            correct[tc] += 1
            total_correct += 1
        if is_enhanced_correct(tlen, pc):
            enhanced_correct[tc] += 1
            total_enhanced_correct += 1

    strict_acc = total_correct / n
    enhanced_acc = total_enhanced_correct / n
    print(f"\n[{name}] 分类整体准确率 (strict): {strict_acc:.4f}")
    print(f"[{name}] Enhanced accuracy: {enhanced_acc:.4f}")

    print("按档位的 strict 召回率：")
    for cls_idx in range(len(BIN_EDGES) - 1):
        s = support.get(cls_idx, 0)
        c = correct.get(cls_idx, 0)
        if s > 0:
            r = c / s
            print(
                f"  Class {cls_idx}: support={s}, correct={c}, recall={r:.4f}")
        else:
            print(f"  Class {cls_idx}: support=0, correct=0, recall=N/A")

    print("按档位的 Enhanced 召回率：")
    for cls_idx in range(len(BIN_EDGES) - 1):
        s = support.get(cls_idx, 0)
        c = enhanced_correct.get(cls_idx, 0)
        if s > 0:
            r = c / s
            print(
                f"  Class {cls_idx}: support={s}, enhanced_correct={c}, enhanced_recall={r:.4f}"
            )
        else:
            print(
                f"  Class {cls_idx}: support=0, enhanced_correct=0, enhanced_recall=N/A"
            )


async def run_eval(concurrency: int) -> None:
    print(f"[INFO] Loading samples from {DATA_PATH} (first {MAX_SAMPLES})...")
    samples = load_samples(DATA_PATH, MAX_SAMPLES)
    print(f"[INFO] Loaded {len(samples)} samples.")
    print(f"[INFO] Using concurrency={concurrency}")

    true_token_counts: List[float] = []
    true_lengths_llama: List[float] = []
    pred_reg: List[float] = []
    pred_cls: List[int] = []

    pred_reg_llama: List[float] = []
    pred_cls_llama: List[int] = []

    preview: List[Dict[str, Any]] = []

    # 用信号量限制并发数
    sem = asyncio.Semaphore(concurrency)

    async def handle_one(idx: int, sample: Dict[str, Any],
                         client: httpx.AsyncClient) -> None:
        nonlocal true_token_counts, true_lengths_llama
        nonlocal pred_reg, pred_cls, pred_reg_llama, pred_cls_llama, preview

        messages = build_chat_messages(sample)

        async with sem:
            resp = await call_openai_chat(client, messages)

        if resp is None:
            return

        usage = resp.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            print(f"[WARN] No completion_tokens for sample {idx}, skip.")
            return

        oracle = resp.get("oracle_metadata") or {}
        cls_idx = oracle.get("class_index")
        reg_len = oracle.get("regression_estimated_length")
        if cls_idx is None or reg_len is None:
            print(f"[WARN] No oracle_metadata for sample {idx}, skip.")
            return

        token_counts = sample.get("token_counts") or {}
        true_out_tokens = float(token_counts.get("output", 0))

        true_token_counts.append(true_out_tokens)
        pred_reg.append(float(reg_len))
        pred_cls.append(int(cls_idx))

        true_lengths_llama.append(float(completion_tokens))
        pred_reg_llama.append(float(reg_len))
        pred_cls_llama.append(int(cls_idx))

        if len(preview) < 5:
            user_text = ""
            for m in messages:
                if m["role"] == "user":
                    user_text = m["content"]
                    break
            preview.append({
                "sample_index": idx,
                "user_prompt_preview":
                user_text[:80].replace("\n", " "),
                "true_len_token_counts": true_out_tokens,
                "true_len_llama": completion_tokens,
                "pred_class": int(cls_idx),
                "pred_reg_len": float(reg_len),
            })

    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [
            handle_one(idx, sample, client)
            for idx, sample in enumerate(samples)
        ]
        await asyncio.gather(*tasks)

    print("\n=== 样例输出预览（前 5 条）===")
    for row in preview:
        print(json.dumps(row, ensure_ascii=False))

    compute_stats(
        name="真值=token_counts.output",
        true_lengths=true_token_counts,
        pred_reg_lengths=pred_reg,
        pred_class_indices=pred_cls,
    )

    compute_stats(
        name="真值=vLLM Llama-3.1-8B-Instruct 生成长度",
        true_lengths=true_lengths_llama,
        pred_reg_lengths=pred_reg_llama,
        pred_class_indices=pred_cls_llama,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline oracle evaluation via vLLM HTTP server.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Maximum number of concurrent HTTP requests (default: 50).",
    )
    args = parser.parse_args()

    asyncio.run(run_eval(args.concurrency))


if __name__ == "__main__":
    main()
