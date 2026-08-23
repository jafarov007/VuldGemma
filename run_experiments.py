import os
import sys
import copy
import argparse
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.utils import load_config, set_seed
from src.data import CodeGraphDataset, PadCollator
from src.model import VuldGemmaModel
from src.train import Trainer
from src.strategies import SeedAggregator


def deep_copy_cfg(cfg):
    """Deep copy config dict (skip non-serializable)."""
    return json.loads(json.dumps(cfg))


def run_single(cfg, seed, tokenizer, train_dataset, valid_dataset):
    """Run a single training experiment and return metrics."""
    set_seed(seed)
    
    mode = cfg.get("ablation_mode", "full")
    tag = cfg.get("_experiment_tag", f"seed{seed}_{mode}")
    cfg["save_dir"] = f"./runs/{tag}"
    
    # Dynamic class weighting
    num_safe = sum(1 for p in train_dataset.pairs if p.label == 0)
    num_vuln = max(1, sum(1 for p in train_dataset.pairs if p.label == 1))
    if "loss" not in cfg:
        cfg["loss"] = {}
    cfg["loss"]["pos_weight"] = num_safe / num_vuln
    
    # DataLoaders
    batch_size = int(cfg.get("batch_size", 8))
    collator = PadCollator()
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        collate_fn=collator, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        collate_fn=collator, pin_memory=True
    )
    
    # Build & Train
    model = VuldGemmaModel(cfg)
    trainer = Trainer(model, train_loader, valid_loader, cfg)
    best_metrics = trainer.run()
    
    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()
    
    return best_metrics


def load_datasets(cfg, tokenizer):
    """Load datasets once, reuse for all experiments."""
    paths_cfg = cfg["paths"]
    train_dataset = CodeGraphDataset(
        paths_cfg=paths_cfg, split="train", tokenizer=tokenizer,
        code_max=cfg["seq_lens"]["code_max"], graph_max=cfg["seq_lens"]["graph_max"]
    )
    valid_dataset = CodeGraphDataset(
        paths_cfg=paths_cfg, split="valid", tokenizer=tokenizer,
        code_max=cfg["seq_lens"]["code_max"], graph_max=cfg["seq_lens"]["graph_max"]
    )
    print(f"  Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
    return train_dataset, valid_dataset


def print_table(title, rows):
    """Pretty-print a results table."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    
    if not rows:
        print("  No results.")
        return
    
    # Header
    keys = list(rows[0].keys())
    header = f"  {'Experiment':<30}"
    for k in keys:
        if k != "experiment":
            header += f" {k:>12}"
    print(header)
    print(f"  {'-'*30}" + f" {'-'*12}" * (len(keys) - 1))
    
    for row in rows:
        line = f"  {row.get('experiment', '?'):<30}"
        for k in keys:
            if k != "experiment":
                v = row[k]
                if isinstance(v, float):
                    line += f" {v:>12.4f}"
                else:
                    line += f" {str(v):>12}"
        print(line)
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="VuldGemma Experiments")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated seeds (e.g., '42,123,456')")
    parser.add_argument("--ablation", action="store_true",
                        help="Modal ablation: code_only, graph_only, full")
    parser.add_argument("--component-ablation", action="store_true",
                        help="Component ablation: EMA on/off, Threshold on/off")
    parser.add_argument("--without-ema", action="store_true", help="Run without EMA")
    parser.add_argument("--without-focal", action="store_true", help="Run without Focal Loss (use BCE)")
    parser.add_argument("--without-threshold", action="store_true", help="Run without Optimal Threshold (use 0.5)")
    args = parser.parse_args()
    
    base_cfg = load_config(args.config)
    
    # Apply individual ablation flags if running a standard experiment
    if args.without_ema:
        base_cfg.setdefault("strategies", {})["ema_enabled"] = False
    if args.without_threshold:
        base_cfg.setdefault("strategies", {})["optimal_threshold"] = False
    if args.without_focal:
        base_cfg.setdefault("loss", {})["type"] = "bce"

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Load tokenizer & datasets once
    model_name = base_cfg.get("model", {}).get("base", "google/gemma-1.1-1b-it")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset, valid_dataset = load_datasets(base_cfg, tokenizer)
    
    all_results = []  # Collect all experiment results
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TABLE A: Modal Ablation (code_only, graph_only, full)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if args.ablation:
        modes = ["code_only", "graph_only", "full"]
    else:
        modes = ["full"]
    
    modal_results = []
    for mode in modes:
        aggregator = SeedAggregator()
        
        for seed in seeds:
            cfg = deep_copy_cfg(base_cfg)
            cfg["ablation_mode"] = mode
            cfg["_experiment_tag"] = f"modal_{mode}_seed{seed}"
            
            print(f"\n{'#'*60}")
            print(f"  MODAL ABLATION: mode={mode}, seed={seed}")
            print(f"{'#'*60}")
            
            metrics = run_single(cfg, seed, tokenizer, train_dataset, valid_dataset)
            aggregator.add_run(seed, metrics)
        
        summary = aggregator.summary()
        modal_results.append({
            "experiment": mode,
            "F1": summary.get("f1_vuln_mean", 0),
            "F1@0.5": summary.get("f1_at_0.5_mean", 0),
            "Precision": summary.get("precision_mean", 0),
            "Recall": summary.get("recall_mean", 0),
            "Threshold": summary.get("best_threshold_mean", 0.5),
        })
        aggregator.save(f"./runs/modal_{mode}_results.json")
    
    if args.ablation:
        print_table("TABLE A: Modal Ablation: ", modal_results)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TABLE B: Component Ablation (EMA, Threshold effects)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if args.component_ablation:
        component_experiments = [
            # (experiment_name, use_ema, use_focal, use_thresh)
            ("Full System (+EMA, +Focal, +Opt.Threshold)", True, True, True),
            ("- w/o Focal Loss (uses BCE)",                True, False, True),
            ("- w/o EMA",                                  False, True, True),
            ("- w/o Optimal Threshold (uses 0.5)",         True, True, False),
        ]
        
        comp_results = []
        for exp_name, ema_on, focal_on, thresh_on in component_experiments:
            aggregator = SeedAggregator()
            
            for seed in seeds:
                cfg = deep_copy_cfg(base_cfg)
                cfg["ablation_mode"] = "full"
                cfg["strategies"] = {
                    "ema_enabled": ema_on,
                    "ema_decay": 0.999,
                    "optimal_threshold": thresh_on,
                }
                cfg.setdefault("loss", {})["type"] = "focal" if focal_on else "bce"
                
                tag = f"comp_ema{int(ema_on)}_focal{int(focal_on)}_thresh{int(thresh_on)}_seed{seed}"
                cfg["_experiment_tag"] = tag
                
                print(f"\n{'#'*60}")
                print(f"  COMPONENT ABLATION: {exp_name}, seed={seed}")
                print(f"{'#'*60}")
                
                metrics = run_single(cfg, seed, tokenizer, train_dataset, valid_dataset)
                aggregator.add_run(seed, metrics)
            
            summary = aggregator.summary()
            comp_results.append({
                "experiment": exp_name,
                "F1": summary.get("f1_vuln_mean", 0),
                "F1@0.5": summary.get("f1_at_0.5_mean", 0),
                "Precision": summary.get("precision_mean", 0),
                "Recall": summary.get("recall_mean", 0),
            })
            tag_base = f"comp_ema{int(ema_on)}_focal{int(focal_on)}_thresh{int(thresh_on)}"
            aggregator.save(f"./runs/{tag_base}_results.json")
        
        print_table("TABLE B: Component Ablation — ALL components impact", comp_results)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TABLE C: Multi-Seed Robustness (if multiple seeds)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if len(seeds) > 1 and not args.ablation and not args.component_ablation:
        # Simple multi-seed run with default config
        aggregator = SeedAggregator()
        for seed in seeds:
            cfg = deep_copy_cfg(base_cfg)
            cfg["_experiment_tag"] = f"robustness_seed{seed}"
            
            print(f"\n{'#'*60}")
            print(f"  MULTI-SEED: seed={seed}")
            print(f"{'#'*60}")
            
            metrics = run_single(cfg, seed, tokenizer, train_dataset, valid_dataset)
            aggregator.add_run(seed, metrics)
        
        aggregator.print_summary()
        aggregator.save("./runs/robustness_results.json")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Save all results
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n\nALL result save in ./runs/ .")
    print("Check the JSON file : ls runs/*.json")


if __name__ == "__main__":
    main()
