import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from typing import Dict, Any, Optional

from .utils import AverageMeter, MetricTracker, Logger, Timer, save_checkpoint, format_time
from .losses import FocalLoss
from .model import VuldGemmaModel
from .strategies import ThresholdOptimizer, ModelEMA


class Trainer:

    def __init__(self, model: VuldGemmaModel, train_loader, valid_loader, cfg: Dict[str, Any]):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.cfg = cfg

        self.device = torch.device(cfg.get("device", "cuda"))
        self.model.to(self.device)

        # ── Optimizer ──
        optim_cfg = cfg.get("optim", {})
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(optim_cfg.get("lr", 3e-4)),
            weight_decay=float(optim_cfg.get("weight_decay", 0.01))
        )

        # ── Scheduler ──
        self.epochs = int(cfg.get("epochs", 8))
        total_steps = len(self.train_loader) * self.epochs
        warmup_steps = int(total_steps * float(optim_cfg.get("warmup_ratio", 0.10)))

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # ── Loss ──
        loss_cfg = cfg.get("loss", {})
        loss_type = loss_cfg.get("type", "focal")
        if loss_type == "focal":
            self.criterion = FocalLoss(
                gamma=float(loss_cfg.get("focal_gamma", 2.0)),
                pos_weight=float(loss_cfg.get("pos_weight", 1.0)),
                label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
                reduction="mean"
            ).to(self.device)
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([float(loss_cfg.get("pos_weight", 1.0))])
            ).to(self.device)

        # ── Training mechanics ──
        precision = cfg.get("precision", "bf16")
        self.use_bf16 = (precision == "bf16")
        self.use_fp16 = (precision == "fp16")
        self.scaler = GradScaler('cuda', enabled=self.use_fp16)
        self.grad_accum_steps = int(cfg.get("grad_accum_steps", 2))
        self.max_grad_norm = float(optim_cfg.get("max_grad_norm", 1.0))

        # ── EMA (Exponential Moving Average) — opsiyonel ──
        strat_cfg = cfg.get("strategies", {})
        self.use_ema = strat_cfg.get("ema_enabled", False)
        if self.use_ema:
            self.ema = ModelEMA(self.model, decay=strat_cfg.get("ema_decay", 0.999))
            print("  [EMA] Active (decay=%.4f)" % strat_cfg.get("ema_decay", 0.999))
        else:
            self.ema = None
            print("  [EMA] Inactive")
        
        self.use_optimal_threshold = strat_cfg.get("optimal_threshold", True)
        self.threshold_max_recall = strat_cfg.get("threshold_max_recall", None)

        # ── Mode (for ablation studies) ──
        self.mode = cfg.get("ablation_mode", "full")

        # ── Logging ──
        self.save_dir = cfg.get("save_dir", "./runs")
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = Logger(use_tb=True, save_dir=self.save_dir)
        self.metric_tracker = MetricTracker()

        self.best_f1 = 0.0
        self.best_epoch = 0
        self.best_threshold = 0.5

    # ────────────────────────────────────────────────
    def _get_amp_dtype(self):
        if self.use_bf16:
            return torch.bfloat16
        elif self.use_fp16:
            return torch.float16
        return torch.float32

    # ────────────────────────────────────────────────
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        loss_meter = AverageMeter()
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            c_ids = batch["code_input_ids"].to(self.device)
            c_mask = batch["code_attention_mask"].to(self.device)
            g_ids = batch["graph_input_ids"].to(self.device)
            g_mask = batch["graph_attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with autocast('cuda', dtype=self._get_amp_dtype()):
                output = self.model(c_ids, c_mask, g_ids, g_mask, mode=self.mode)
                logits = output["logits"]
                loss = self.criterion(logits, labels)
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()
            loss_meter.update(loss.item() * self.grad_accum_steps)

            # Gradient accumulation step
            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update EMA after each optimizer step
                if self.use_ema:
                    self.ema.update(self.model)

            if step % 10 == 0:
                lr = self.scheduler.get_last_lr()[0]
                self.logger.log(
                    f"  [{epoch}][{step:>4d}/{len(self.train_loader)}] "
                    f"loss={loss_meter.avg():.4f} lr={lr:.2e}"
                )

        return loss_meter.avg()

    # ────────────────────────────────────────────────
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        # Apply EMA weights for evaluation (if enabled)
        if self.use_ema:
            self.ema.apply_shadow(self.model)
        self.model.eval()
        
        loss_meter = AverageMeter()
        all_probs = []
        all_labels = []

        for batch in self.valid_loader:
            c_ids = batch["code_input_ids"].to(self.device)
            c_mask = batch["code_attention_mask"].to(self.device)
            g_ids = batch["graph_input_ids"].to(self.device)
            g_mask = batch["graph_attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with autocast('cuda', dtype=self._get_amp_dtype()):
                output = self.model(c_ids, c_mask, g_ids, g_mask, mode=self.mode)
                logits = output["logits"]
                loss = self.criterion(logits, labels)

            loss_meter.update(loss.item())

            # Convert logits to probabilities
            if logits.shape[-1] == 2:
                probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            else:
                probs = torch.sigmoid(logits.float()).squeeze(-1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy().flatten())

        # Restore original weights
        if self.use_ema:
            self.ema.restore(self.model)

        # Concatenate all predictions
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels).astype(int)

        # ── Optimal Threshold Search ──
        if self.use_optimal_threshold:
            best_t, best_f1_at_t = ThresholdOptimizer.find_optimal_threshold(
                all_probs, all_labels, max_recall=self.threshold_max_recall
            )
        else:
            best_t = 0.5
            preds_t = (all_probs >= 0.5).astype(int)
            tp_t = np.sum((preds_t == 1) & (all_labels == 1))
            fp_t = np.sum((preds_t == 1) & (all_labels == 0))
            fn_t = np.sum((preds_t == 0) & (all_labels == 1))
            best_f1_at_t = 2 * tp_t / (2 * tp_t + fp_t + fn_t + 1e-8)
        
        # Compute all metrics at optimal threshold
        preds_bin = (all_probs >= best_t).astype(int)
        tp = np.sum((preds_bin == 1) & (all_labels == 1))
        fp = np.sum((preds_bin == 1) & (all_labels == 0))
        fn = np.sum((preds_bin == 0) & (all_labels == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        # Also compute at fixed 0.5 for comparison
        preds_05 = (all_probs >= 0.5).astype(int)
        tp_05 = np.sum((preds_05 == 1) & (all_labels == 1))
        fp_05 = np.sum((preds_05 == 1) & (all_labels == 0))
        fn_05 = np.sum((preds_05 == 0) & (all_labels == 1))
        f1_at_05 = 2 * tp_05 / (2 * tp_05 + fp_05 + fn_05 + 1e-8)

        metrics = {
            "val_loss": loss_meter.avg(),
            "f1_vuln": best_f1_at_t,           # F1 at optimal threshold
            "f1_at_0.5": f1_at_05,              # F1 at fixed 0.5 (for comparison)
            "precision": float(precision),
            "recall": float(recall),
            "best_threshold": best_t,
        }

        self.logger.log(
            f"  ── Validation Epoch {epoch} (EMA) ──\n"
            f"     Loss={metrics['val_loss']:.4f} | "
            f"F1={metrics['f1_vuln']:.4f} (t={best_t:.2f}) | "
            f"F1@0.5={f1_at_05:.4f} | "
            f"Prec={metrics['precision']:.4f} | "
            f"Rec={metrics['recall']:.4f}"
        )

        # Log to TensorBoard
        self.logger.log_dict({
            "val/loss": metrics["val_loss"],
            "val/f1_vuln": metrics["f1_vuln"],
            "val/f1_at_0.5": metrics["f1_at_0.5"],
            "val/precision": metrics["precision"],
            "val/recall": metrics["recall"],
            "val/best_threshold": metrics["best_threshold"],
        }, step=epoch)

        return metrics

    # ────────────────────────────────────────────────
    def run(self) -> Dict[str, float]:
        """Main training loop. Returns best metrics."""
        self.logger.log(f"\n{'='*60}")
        self.logger.log(f"  VuldGemma Training — Mode: {self.mode}")
        self.logger.log(f"  Epochs: {self.epochs} | Batch: {self.cfg.get('batch_size')} x {self.grad_accum_steps} accum")
        self.logger.log(f"  Precision: {'bf16' if self.use_bf16 else 'fp16' if self.use_fp16 else 'fp32'}")
        self.logger.log(f"  Strategies: EMA + Optimal Threshold + Focal Loss")
        self.logger.log(f"{'='*60}\n")

        timer = Timer()
        best_metrics = {}

        for epoch in range(1, self.epochs + 1):
            timer.start()

            self.logger.log(f"\n── Epoch {epoch}/{self.epochs} ──")
            train_loss = self.train_epoch(epoch)
            metrics = self.validate(epoch)

            elapsed = timer.stop().elapsed()
            self.logger.log(f"  Time: {format_time(elapsed)} | Train Loss: {train_loss:.4f}")
            self.logger.log_scalar("train/loss", train_loss, epoch)

            # Save best model
            f1_optimal = metrics["f1_vuln"]
            f1_at_05 = metrics.get("f1_at_0.5", 0.0)
            
            # Combined F1 Metric for Robust Model Selection
            combined_score = (f1_optimal * 0.7) + (f1_at_05 * 0.3)
            
            if not hasattr(self, "best_combined"):
                self.best_combined = 0.0
                
            if combined_score > self.best_combined:
                self.best_combined = combined_score
                self.best_f1 = f1_optimal
                self.best_epoch = epoch
                self.best_threshold = metrics["best_threshold"]
                best_metrics = metrics.copy()
                
                # Save with EMA weights (if enabled)
                if self.use_ema:
                    self.ema.apply_shadow(self.model)
                save_path = os.path.join(self.save_dir, f"best_model_ep{epoch}.pt")
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch, save_path,
                    {"f1": f1_optimal, "threshold": self.best_threshold, "mode": self.mode, **metrics}
                )
                if self.use_ema:
                    self.ema.restore(self.model)
                
                self.logger.log(f"  ★ New best F1={self.best_f1:.4f} (t={self.best_threshold:.2f}) at epoch {epoch}")

        self.logger.log(f"\n{'='*60}")
        self.logger.log(f"  Training complete!")
        self.logger.log(f"  Best F1={self.best_f1:.4f} (threshold={self.best_threshold:.2f}) at epoch {self.best_epoch}")
        self.logger.log(f"{'='*60}")
        self.logger.close()
        
        return best_metrics

    # ────────────────────────────────────────────────
    @torch.no_grad()
    def extract_attention_heatmaps(self, dataloader, max_samples: int = 50) -> list:
        if self.use_ema:
            self.ema.apply_shadow(self.model)
        self.model.eval()
        results = []

        for batch in dataloader:
            if len(results) >= max_samples:
                break

            c_ids = batch["code_input_ids"].to(self.device)
            c_mask = batch["code_attention_mask"].to(self.device)
            g_ids = batch["graph_input_ids"].to(self.device)
            g_mask = batch["graph_attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            output = self.model(c_ids, c_mask, g_ids, g_mask, return_attention=True)

            logits = output["logits"]
            if logits.shape[-1] == 2:
                probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            else:
                probs = torch.sigmoid(logits.float()).squeeze(-1)

            preds = (probs > self.best_threshold).long()

            for i in range(c_ids.size(0)):
                if len(results) >= max_samples:
                    break

                entry = {
                    "label": labels[i].item(),
                    "pred": preds[i].item(),
                    "prob": probs[i].item(),
                    "code_tokens": c_ids[i].cpu(),
                    "graph_tokens": g_ids[i].cpu(),
                }

                if "attn_c2g" in output:
                    entry["attn_c2g"] = output["attn_c2g"][i].cpu()
                if "attn_g2c" in output:
                    entry["attn_g2c"] = output["attn_g2c"][i].cpu()

                results.append(entry)

        if self.use_ema:
            self.ema.restore(self.model)
        return results
