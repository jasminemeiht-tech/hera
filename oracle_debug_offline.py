import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

# 调用训练脚本里的实现
ORACLE_DIR = Path("/home/ubuntu/ATHENA/ORACLE").resolve()
import sys

if str(ORACLE_DIR) not in sys.path:
    sys.path.append(str(ORACLE_DIR))

from len_pred import (  # type: ignore  # noqa: E402
    EnhancedV7EightClassDataset,
    EnhancedV7ModelRebuilder,
    compute_enhanced_v7_classification_metrics,
)


JSONL_PATH = Path(
    "/home/ubuntu/ATHENA/DATASET/dataset_served_alpaca/"
    "batch-01_20251113_180031/train/fold_1.jsonl"
)
CKPT_PATH = Path(
    "/home/ubuntu/ATHENA/ORACLE/__pycache__/full_run_train_batch_3/"
    "len_pred_full.pth"
)
MAX_SAMPLES = 500
BATCH_SIZE = 16


def run_offline_eval() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading checkpoint from: {CKPT_PATH}")

    # 1) 从 best checkpoint 重建系统
    system = EnhancedV7ModelRebuilder.rebuild_from_checkpoint(
        str(CKPT_PATH),
        device=str(device),
        verify_functionality=False,
        restore_training_state=False,
    )

    tokenizer = system.feature_extractor.tokenizer
    discretizer = system.discretizer

    # 2) 用训练时的 Dataset 实现加载 fold_1.jsonl
    print(f"[INFO] Building EnhancedV7EightClassDataset from {JSONL_PATH}")
    dataset = EnhancedV7EightClassDataset(
        jsonl_files=[str(JSONL_PATH)],
        tokenizer=tokenizer,
        discretizer=discretizer,
        max_length=512,
        boundary_sample_weight=1.2,
        use_soft_labels=False,
    )
    print(f"[INFO] Dataset size (all folds in this file): {len(dataset)}")

    data_collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=data_collator,
    )

    true_lengths: List[float] = []
    pred_classes: List[int] = []
    class_probs: List[List[float]] = []

    # 3) 逐 batch 运行与训练相同的前向路径
    system.feature_extractor.set_training_mode(False)
    system.predictor.eval()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            original_lengths = batch["original_length"].to(device)
            prompt_lengths = batch["prompt_length"].to(device)

            hidden_states = system.feature_extractor.extract_features(
                input_ids, attention_mask, training=False
            )

            class_logits, _ = system.predictor(
                hidden_states, attention_mask, prompt_lengths
            )

            probs = torch.softmax(class_logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            true_lengths.extend(original_lengths.cpu().tolist())
            pred_classes.extend(preds.cpu().tolist())
            class_probs.extend(probs.cpu().tolist())

            if len(true_lengths) >= MAX_SAMPLES:
                break

    # 截取前 MAX_SAMPLES
    true_lengths_np = np.array(true_lengths[:MAX_SAMPLES], dtype=float)
    pred_classes_np = np.array(pred_classes[:MAX_SAMPLES], dtype=int)
    class_probs_np = np.array(class_probs[:MAX_SAMPLES], dtype=float)

    print(f"[INFO] Collected {len(true_lengths_np)} samples for offline eval.")

    # 4) 使用训练时同一个 metrics 函数计算指标
    metrics = compute_enhanced_v7_classification_metrics(
        y_true_lengths=true_lengths_np,
        y_pred_classes=pred_classes_np,
        y_prob=class_probs_np,
        discretizer=discretizer,
        stage_name="OfflineDebug",
    )

    print("\n=== Offline Debug Metrics on fold_1 (first "
          f"{len(true_lengths_np)} samples) ===")
    print(f"overall_accuracy (strict)   : {metrics['overall_accuracy']:.4f}")
    print(f"enhanced_accuracy           : {metrics['enhanced_accuracy']:.4f}")
    print(f"avg_class_distance          : {metrics['avg_class_distance']:.4f}")
    print(f"within_1_accuracy           : {metrics['within_1_accuracy']:.4f}")
    print(f"boundary_aware_accuracy     : {metrics['boundary_aware_accuracy']:.4f}")
    print(f"num_samples                 : {metrics['num_samples']}")


if __name__ == "__main__":
    run_offline_eval()
