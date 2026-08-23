"""
VuldGemma — Training Entry Point
==================================
Usage:
    python run_train.py --config config.yaml
    
    # Ablation studies:
    python run_train.py --config config.yaml --mode code_only
    python run_train.py --config config.yaml --mode graph_only
    python run_train.py --config config.yaml --mode full
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.utils import load_config, set_seed
from src.data import CodeGraphDataset, PadCollator
from src.model import VuldGemmaModel
from src.train import Trainer


def main():
    parser = argparse.ArgumentParser(description="VuldGemma Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", type=str, default=None,
                        help="Ablation mode: 'full' (default), 'code_only', 'graph_only'")
    parser.add_argument("--without-ema", action="store_true", help="Disable EMA for this run")
    parser.add_argument("--without-focal", action="store_true", help="Disable Focal Loss (use BCE) for this run")
    parser.add_argument("--without-threshold", action="store_true", help="Disable Optimal Threshold (use 0.5) for this run")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    # Override ablation mode if passed via CLI
    if args.mode:
        cfg["ablation_mode"] = args.mode
    
    mode = cfg.get("ablation_mode", "full")
    
    # Override components if flags are passed
    if args.without_ema:
        cfg.setdefault("strategies", {})["ema_enabled"] = False
        print("  [Ablation] EMA Disabled.")
    if args.without_threshold:
        cfg.setdefault("strategies", {})["optimal_threshold"] = False
        print("  [Ablation] Optimal Threshold Disabled (will use 0.5).")
    if args.without_focal:
        cfg.setdefault("loss", {})["type"] = "bce"
        print("  [Ablation] Focal Loss Disabled (using BCEWithLogitsLoss).")
        
    print(f"\n[{cfg['project']}] Starting setup... (mode={mode})")

    # Load Tokenizer
    model_name = cfg.get("model", {}).get("base", "google/gemma-1.1-1b-it")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Tokenizer: {model_name}")

    # Build Datasets
    paths_cfg = cfg["paths"]
    train_dataset = CodeGraphDataset(
        paths_cfg=paths_cfg,
        split="train",
        tokenizer=tokenizer,
        code_max=cfg["seq_lens"]["code_max"],
        graph_max=cfg["seq_lens"]["graph_max"]
    )

    valid_dataset = CodeGraphDataset(
        paths_cfg=paths_cfg,
        split="valid",
        tokenizer=tokenizer,
        code_max=cfg["seq_lens"]["code_max"],
        graph_max=cfg["seq_lens"]["graph_max"]
    )

    print(f"  Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")

    # ── Dynamic Class Weighting ──
    num_safe = sum(1 for p in train_dataset.pairs if p.label == 0)
    num_vuln = sum(1 for p in train_dataset.pairs if p.label == 1)
    num_vuln = max(1, num_vuln)
    num_safe = max(1, num_safe)
    dynamic_pos_weight = num_safe / num_vuln

    if "loss" not in cfg:
        cfg["loss"] = {}

    use_dynamic = cfg.get("loss", {}).get("use_dynamic_pos_weight", True)
    if use_dynamic:
        cfg["loss"]["pos_weight"] = dynamic_pos_weight
        print(f"  [Class Balance] Safe={num_safe}, Vuln={num_vuln} → pos_weight={dynamic_pos_weight:.2f} (DYNAMIC)")
    else:
        cfg["loss"]["pos_weight"] = cfg.get("loss", {}).get("pos_weight", 1.0)
        print(f"  [Class Balance] Safe={num_safe}, Vuln={num_vuln} → pos_weight={cfg['loss']['pos_weight']:.2f} (FIXED — Focal Loss handles imbalance)")

    # Build DataLoaders
    batch_size = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 4))
    collator = PadCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    # Build Model
    print(f"\n  Loading model: {model_name}")
    model = VuldGemmaModel(cfg)

    # Start Training
    trainer = Trainer(model, train_loader, valid_loader, cfg)
    trainer.run()


if __name__ == "__main__":
    main()
