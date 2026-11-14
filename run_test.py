import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from transformers import AutoTokenizer

from src.utils import load_config, set_seed
from src.data import CodeGraphDataset, PadCollator
from src.model import VuldGemmaModel
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score


def main():
    parser = argparse.ArgumentParser(description="VuldGemma Test Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (.pt file)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed threshold. If not set, uses validation threshold from checkpoint.")
    parser.add_argument("--search-test", action="store_true", default=False,
                        help="Search optimal threshold on test set. "
                             "WARNING: This is data leakage — use only for analysis purposes.")
    parser.add_argument("--mode", type=str, default=None,
                        help="Ablation mode override: 'full', 'code_only', 'graph_only'")
    args = parser.parse_args()

    # ── Config ──
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    
    if args.mode:
        cfg["ablation_mode"] = args.mode
        
    mode = cfg.get("ablation_mode", "full")
    device = torch.device(cfg.get("device", "cuda"))
    precision = cfg.get("precision", "bf16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    # ── Tokenizer ──
    model_name = cfg["model"]["base"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Test Dataset ──
    paths_cfg = cfg["paths"]
    test_dataset = CodeGraphDataset(
        paths_cfg=paths_cfg,
        split="test",
        tokenizer=tokenizer,
        code_max=cfg["seq_lens"]["code_max"],
        graph_max=cfg["seq_lens"]["graph_max"]
    )

    batch_size = int(cfg.get("batch_size", 32))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        collate_fn=PadCollator(),
        pin_memory=True
    )

    num_safe = sum(1 for p in test_dataset.pairs if p.label == 0)
    num_vuln = sum(1 for p in test_dataset.pairs if p.label == 1)
    print(f"\n{'='*60}")
    print(f"  VuldGemma Test Evaluation")
    print(f"{'='*60}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Mode       : {mode}")
    print(f"  Test samples: {len(test_dataset)} (Safe={num_safe}, Vuln={num_vuln})")
    print(f"{'='*60}\n")

    # ── Model ──
    model = VuldGemmaModel(cfg)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
        meta = ckpt.get("meta", {})
        saved_threshold = meta.get("threshold", ckpt.get("best_threshold", 0.5))
        saved_f1 = meta.get("f1", None)
        saved_epoch = ckpt.get("epoch", "?")
        print(f"  Loaded checkpoint from epoch {saved_epoch}")
        if saved_f1:
            print(f"  Validation F1 at save: {saved_f1:.4f} (t={saved_threshold:.2f})")
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        saved_threshold = ckpt.get("best_threshold", 0.5)
        saved_f1 = ckpt.get("best_f1", None)
        saved_epoch = ckpt.get("epoch", "?")
        print(f"  Loaded checkpoint from epoch {saved_epoch}")
    else:
        # Simple state dict
        model.load_state_dict(ckpt, strict=False)
        saved_threshold = 0.5
        saved_epoch = "?"

    model = model.to(device)
    model.eval()

    # ── Inference ──
    all_probs = []
    all_labels = []

    print("  Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            c_ids = batch["code_input_ids"].to(device)
            c_mask = batch["code_attention_mask"].to(device)
            g_ids = batch["graph_input_ids"].to(device)
            g_mask = batch["graph_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast('cuda', dtype=amp_dtype):
                output = model(c_ids, c_mask, g_ids, g_mask, mode=mode)
                logits = output["logits"]

            if logits.shape[-1] == 2:
                probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            else:
                probs = torch.sigmoid(logits.float()).squeeze(-1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy().flatten())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels).astype(int)

    # ── Threshold Selection ──
    if args.threshold is not None:
        # User explicitly provided a threshold
        best_t = args.threshold
        print(f"\n  Using user-specified threshold: {best_t:.2f}")
    elif args.search_test:
        # Explicitly requested: search optimal on test set (DATA LEAKAGE — for analysis only)
        from src.strategies import ThresholdOptimizer
        best_t, best_f1_search = ThresholdOptimizer.find_optimal_threshold(
            all_probs, all_labels,
            max_recall=cfg.get("strategies", {}).get("threshold_max_recall", None)
        )
        print(f"\n  [SEARCH-TEST] Optimal threshold on test set: {best_t:.2f} "
              f"(F1={best_f1_search:.4f})")
        print(f"  ⚠ WARNING: This is data leakage — use only for analysis/comparison!")
    else:
        # Default: use validation threshold from checkpoint (CORRECT methodology)
        best_t = saved_threshold
        print(f"\n  Using validation threshold from checkpoint: {best_t:.2f}")

    # ── Compute Metrics ──
    preds_bin = (all_probs >= best_t).astype(int)
    tp = np.sum((preds_bin == 1) & (all_labels == 1))
    fp = np.sum((preds_bin == 1) & (all_labels == 0))
    fn = np.sum((preds_bin == 0) & (all_labels == 1))
    tn = np.sum((preds_bin == 0) & (all_labels == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    # Sklearn metrics
    f2 = fbeta_score(all_labels, preds_bin, beta=2.0, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.0
        pr_auc = 0.0

    # F1 at fixed 0.5
    preds_05 = (all_probs >= 0.5).astype(int)
    tp_05 = np.sum((preds_05 == 1) & (all_labels == 1))
    fp_05 = np.sum((preds_05 == 1) & (all_labels == 0))
    fn_05 = np.sum((preds_05 == 0) & (all_labels == 1))
    tn_05 = np.sum((preds_05 == 0) & (all_labels == 0))
    f1_05 = 2 * tp_05 / (2 * tp_05 + fp_05 + fn_05 + 1e-8)
    prec_05 = tp_05 / (tp_05 + fp_05 + 1e-8)
    rec_05 = tp_05 / (tp_05 + fn_05 + 1e-8)
    acc_05 = (tp_05 + tn_05) / (tp_05 + tn_05 + fp_05 + fn_05 + 1e-8)

    # ── Print Results ──
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS (mode={mode})")
    print(f"{'='*60}")
    print(f"  Threshold (optimal) : {best_t:.2f}")
    print(f"  F1 (vuln class)     : {f1:.4f}")
    print(f"  F2 (vuln class)     : {f2:.4f}")
    print(f"  Precision           : {precision:.4f}")
    print(f"  Recall              : {recall:.4f}")
    print(f"  Accuracy            : {accuracy:.4f}")
    print(f"  ROC-AUC             : {roc_auc:.4f}")
    print(f"  PR-AUC              : {pr_auc:.4f}")
    print(f"  ──────────────────────────────")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"{'='*60}")
    print(f"\n  --- Results @ Fixed Threshold 0.50 ---")
    print(f"  F1 @ 0.5            : {f1_05:.4f}")
    print(f"  Precision @ 0.5     : {prec_05:.4f}")
    print(f"  Recall @ 0.5        : {rec_05:.4f}")
    print(f"  Accuracy @ 0.5      : {acc_05:.4f}")
    print(f"  ──────────────────────────────")
    print(f"  TP={tp_05}  FP={fp_05}  FN={fn_05}  TN={tn_05}")
    print(f"{'='*60}")

    # ── Probability Distribution ──
    vuln_probs = all_probs[all_labels == 1]
    safe_probs = all_probs[all_labels == 0]
    print(f"\n  Probability Distribution:")
    print(f"    Vulnerable samples: mean={vuln_probs.mean():.4f}, median={np.median(vuln_probs):.4f}")
    print(f"    Safe samples      : mean={safe_probs.mean():.4f}, median={np.median(safe_probs):.4f}")
    print(f"    Separation gap    : {vuln_probs.mean() - safe_probs.mean():.4f}")

    # ── Save results ──
    results = {
        "checkpoint": args.checkpoint,
        "mode": mode,
        "threshold": float(best_t),
        "f1_vuln": float(f1),
        "f2_vuln": float(f2),
        "f1_at_05": float(f1_05),
        "precision": float(precision),
        "recall": float(recall),
        "precision_at_05": float(prec_05),
        "recall_at_05": float(rec_05),
        "accuracy": float(accuracy),
        "accuracy_at_05": float(acc_05),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "tp_05": int(tp_05), "fp_05": int(fp_05), "fn_05": int(fn_05), "tn_05": int(tn_05),
        "vuln_prob_mean": float(vuln_probs.mean()),
        "safe_prob_mean": float(safe_probs.mean()),
    }

    out_path = os.path.join(os.path.dirname(args.checkpoint), "test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    import json
    main()
