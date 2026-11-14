import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


# ============================================================
# 1) Optimal Threshold Finder
# ============================================================

class ThresholdOptimizer:
    @staticmethod
    def find_optimal_threshold(
        probs: np.ndarray, 
        labels: np.ndarray, 
        thresholds: np.ndarray = None,
        max_recall: Optional[float] = None
    ) -> Tuple[float, float]:
        """Search the threshold that maximizes F1.

        If ``max_recall`` is given, thresholds whose recall exceeds it are skipped
        (this constrained search was used for the Devign experiments, with 0.80).
        """

        if thresholds is None:
            thresholds = np.arange(0.05, 0.96, 0.01)
        
        best_f1 = 0.0
        best_t = 0.5
        
        for t in thresholds:
            preds = (probs >= t).astype(int)
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

            if max_recall is not None and recall > max_recall:
                continue

            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        # Fallback: no threshold satisfied the recall constraint -> most conservative threshold
        if max_recall is not None and best_f1 == 0.0:
            best_t = float(thresholds[-1])
            preds = (probs >= best_t).astype(int)
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            best_f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

        return float(best_t), float(best_f1)


# ============================================================
# 2) Exponential Moving Average (EMA)
# ============================================================

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # Deep copy of model parameters
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Call after each optimizer step."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1.0 - self.decay) * param.data
                )
    
    def apply_shadow(self, model: nn.Module):
        """Replace model weights with EMA weights (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self, model: nn.Module):
        """Restore original weights (after evaluation)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}


# ============================================================
# 3) Multi-Seed Results Aggregator
# ============================================================

class SeedAggregator:
    def __init__(self):
        self.results: List[Dict[str, float]] = []
        self.seeds: List[int] = []
    
    def add_run(self, seed: int, metrics: Dict[str, float]):
        self.seeds.append(seed)
        self.results.append(metrics)
    
    def summary(self) -> Dict[str, str]:
        """Returns mean ± std for each metric."""
        if not self.results:
            return {}
        
        keys = self.results[0].keys()
        summary = {}
        
        for k in keys:
            values = [r[k] for r in self.results if k in r]
            if values:
                mean = np.mean(values)
                std = np.std(values)
                summary[k] = f"{mean:.4f} ± {std:.4f}"
                summary[f"{k}_mean"] = float(mean)
                summary[f"{k}_std"] = float(std)
        
        return summary
    
    def save(self, path: str):
        """Save aggregated results to JSON."""
        data = {
            "seeds": self.seeds,
            "per_seed_results": self.results,
            "aggregated": self.summary()
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[SeedAggregator] Saved to {path}")
    
    def print_summary(self):
        s = self.summary()
        print("\n" + "=" * 50)
        print("  Multi-Seed Results (mean ± std)")
        print("=" * 50)
        for k, v in s.items():
            if isinstance(v, str):
                print(f"  {k:>15s} : {v}")
        print("=" * 50)
