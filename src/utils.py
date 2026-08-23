import os
import re
import json
import math
import time
import random
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# 1) AverageMeter
# ============================================================

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the counters."""
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg_val = 0.0

    def update(self, val: float, n: int = 1):
        """
        Updates the value.

        Args:
            val: New measurement value.
            n: Number of samples represented by this value.
        """
        self.val = float(val)
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg_val = self.sum / self.count

    def avg(self) -> float:
        """Returns the average up to this point."""
        return float(self.avg_val)

    def __str__(self):
        return f"{self.avg():.4f}"


# ============================================================
# 2) MetricTracker
# ============================================================

class MetricTracker:
    """
    Collectively tracks metrics like F1, Precision, Recall, and PR-AUC.
    Specifically optimized for binary classification.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all metric counters."""
        self.preds: List[int] = []
        self.labels: List[int] = []
        self.f1_vuln: float = 0.0
        self.precision: float = 0.0
        self.recall: float = 0.0
        self.pr_auc: float = 0.0

    # --------------------------------------------------------
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Collects batch-level predictions and labels.
        Args:
            preds: Model output predictions.
            labels: True labels.
        """
        if preds.ndim > 1:
            preds = preds.view(-1)
        if labels.ndim > 1:
            labels = labels.view(-1)

        preds = preds.detach().cpu().numpy().astype(float)
        labels = labels.detach().cpu().numpy().astype(int)

        self.preds.extend(preds.tolist())
        self.labels.extend(labels.tolist())

    # --------------------------------------------------------
    def compute_f1_vuln(self) -> float:
        """
        Calculates F1 score for Class-1.
        """
        preds_bin = np.round(self.preds)
        tp = np.sum((preds_bin == 1) & (np.array(self.labels) == 1))
        fp = np.sum((preds_bin == 1) & (np.array(self.labels) == 0))
        fn = np.sum((preds_bin == 0) & (np.array(self.labels) == 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        self.f1_vuln = float(f1)
        self.precision = float(precision)
        self.recall = float(recall)
        return self.f1_vuln

    # --------------------------------------------------------
    def compute_precision(self) -> float:
        """Calculates precision for Class-1."""
        if not self.preds:
            return 0.0
        preds_bin = np.round(self.preds)
        tp = np.sum((preds_bin == 1) & (np.array(self.labels) == 1))
        fp = np.sum((preds_bin == 1) & (np.array(self.labels) == 0))
        self.precision = tp / (tp + fp + 1e-8)
        return float(self.precision)

    # --------------------------------------------------------
    def compute_recall(self) -> float:
        """Calculates recall for Class-1."""
        if not self.preds:
            return 0.0
        preds_bin = np.round(self.preds)
        tp = np.sum((preds_bin == 1) & (np.array(self.labels) == 1))
        fn = np.sum((preds_bin == 0) & (np.array(self.labels) == 1))
        self.recall = tp / (tp + fn + 1e-8)
        return float(self.recall)

    # --------------------------------------------------------
    def compute_pr_auc(self) -> float:
        """
        Calculates PR-AUC.
        Assumes predictions are probabilities.
        """
        try:
            from sklearn.metrics import precision_recall_curve, auc
            precision, recall, _ = precision_recall_curve(self.labels, self.preds)
            self.pr_auc = float(auc(recall, precision))
        except Exception:
            self.pr_auc = 0.0
        return self.pr_auc

    # --------------------------------------------------------
    def summary(self) -> Dict[str, float]:
        """
        Calculates and returns all metrics.
        """
        if not self.preds:
            return {"f1_vuln": 0.0, "precision": 0.0, "recall": 0.0, "pr_auc": 0.0}

        self.compute_f1_vuln()
        self.compute_precision()
        self.compute_recall()
        self.compute_pr_auc()

        return {
            "f1_vuln": self.f1_vuln,
            "precision": self.precision,
            "recall": self.recall,
            "pr_auc": self.pr_auc,
        }

    def __str__(self):
        return (
            f"F1(vuln)={self.f1_vuln:.4f} | "
            f"Prec={self.precision:.4f} | "
            f"Rec={self.recall:.4f} | "
            f"PR-AUC={self.pr_auc:.4f}"
        )
# ============================================================
# 3) Logger
# ============================================================

class Logger:

    def __init__(self, use_tb: bool = True, save_dir: str = "./runs"):
        self.use_tb = bool(use_tb)
        self.save_dir = os.path.normpath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        # TensorBoard writer (optional)
        self.tb: Optional[SummaryWriter] = None
        if self.use_tb:
            # You might want to write each run to a timestamped subfolder
            # but a single folder per project is also possible.
            time_tag = time.strftime("%Y%m%d-%H%M%S")
            tb_dir = os.path.join(self.save_dir, f"tb-{time_tag}")
            os.makedirs(tb_dir, exist_ok=True)
            self.tb = SummaryWriter(log_dir=tb_dir)

        # A simple plain text log file
        self.txt_log_path = os.path.join(self.save_dir, "train.log")
        if not os.path.exists(self.txt_log_path):
            with open(self.txt_log_path, "w", encoding="utf-8") as f:
                f.write("# VuldGemma Training Log\n")


    def log(self, msg: str, also_print: bool = True, step: int | None = None, tag: str = "info"):
      if also_print:
          print(msg)

      # txt log
      try:
          with open(self.txt_log_path, "a", encoding="utf-8") as f:
              f.write(msg + "\n")
      except Exception:
          pass

      # TensorBoard'a text (opsiyonel)
      if self.tb is not None and step is not None:
          try:
              # add_text dislikes empty lines; making it safe
              safe = msg if msg.strip() else "<empty>"
              self.tb.add_text(tag, safe, global_step=int(step))
          except Exception:
              pass


    # --------------------------------------------------------
    def log_scalar(self, name: str, value: float, step: int):
        """Writes a single numeric metric."""
        if self.tb is not None:
            try:
                self.tb.add_scalar(name, float(value), global_step=int(step))
            except Exception:
                pass  # Do not reflect TensorBoard error to training flow

        # txt log (hafif)
        try:
            with open(self.txt_log_path, "a", encoding="utf-8") as f:
                f.write(f"[{step}] {name}={value}\n")
        except Exception:
            pass

    # --------------------------------------------------------
    def log_dict(self, metrics: Dict[str, float], step: int):
        """Writes multiple metrics at once."""
        if not isinstance(metrics, dict):
            return
        # TensorBoard
        if self.tb is not None:
            for k, v in metrics.items():
                try:
                    self.tb.add_scalar(k, float(v), global_step=int(step))
                except Exception:
                    pass

        # txt log
        try:
            with open(self.txt_log_path, "a", encoding="utf-8") as f:
                kv = " ".join([f"{k}={v}" for k, v in metrics.items()])
                f.write(f"[{step}] {kv}\n")
        except Exception:
            pass

    # --------------------------------------------------------
    def save_json(self, data: Dict[str, Any], path: str):
        """
        Saves JSON file. Automatically creates folder.
        """
        os.makedirs(os.path.dirname(os.path.normpath(path)), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    # --------------------------------------------------------
    def print_step(self, epoch: int, step: int, metrics: Dict[str, Any]):
        """
        Prints summary line to console.
        """
        if not isinstance(metrics, dict):
            metrics = {}
        # Sort metrics by key, keep short and readable
        parts = [f"epoch={epoch}", f"step={step}"]
        for k in sorted(metrics.keys()):
            v = metrics[k]
            try:
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            except Exception:
                parts.append(f"{k}={v}")
        line = " | ".join(parts)
        print(line, flush=True)

    # --------------------------------------------------------
    def close(self):
        """Closes resources."""
        if self.tb is not None:
            try:
                self.tb.flush()
                self.tb.close()
            except Exception:
                pass
            self.tb = None

    # For safe exit
    def __del__(self):
        self.close()


# ============================================================
# 4) Timer
# ============================================================

class Timer:
    """
    Simple context manager to measure code block duration.

    Usage:
        with Timer() as t:
            ... heavy work ...
        print("elapsed time:", t.elapsed())

    veya manuel:
        t = Timer().start()
        ... work ...
        t.stop()
        print(t.elapsed())
    """

    def __init__(self):
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    # Context manager girwork
    def __enter__(self):
        self._start = time.perf_counter()
        self._end = None
        return self

    # Context manager exit
    def __exit__(self, *args):
        self._end = time.perf_counter()

    # Manual start
    def start(self):
        self._start = time.perf_counter()
        self._end = None
        return self

    # Manuel durdur
    def stop(self):
        if self._start is None:
            # treat as start/stop if never started
            self._start = time.perf_counter()
        self._end = time.perf_counter()
        return self

    # Elapsed time (seconds)
    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.perf_counter()
        return float(end - self._start)

# ============================================================
# Fonksiyon 1/14 — set_seed
# ============================================================

def set_seed(seed: int) -> None:
    """
    Fixes randomness sources for fully deterministic (as much as possible) run.
    Seed is set for PyTorch, NumPy, and Python `random`.
    CUDA and cuDNN deterministic flags are set.

    Not:
      - Performance might drop if some GPU kernels don't find a deterministic equivalent.
      - Calls `torch.use_deterministic_algorithms` in repro mode.
    """
    if seed is None:
        return
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU

    # Deterministic behavior (at partial performance cost)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

    # cuDNN settings
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
# ============================================================
# Fonksiyon 2/14 — setup_device
# ============================================================

def setup_device(device_str: str) -> torch.device:
    """
    Selects device and returns torch.device.
    Örnekler: "cuda", "cuda:0", "cpu", "mps"

    Behavior:
      - "cuda" istenir ve yoksa otomatik "cpu"ya düşer.
      - "mps" (Apple Silicon) istenir ve destek yoksa "cpu"ya düşer.
      - "cuda:N" istenir ve o indeks mevcut değilse "cpu"ya düşer.
    """
    if not isinstance(device_str, str):
        return torch.device("cpu")

    dev = device_str.lower().strip()

    # CUDA ailesi
    if dev.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        # "cuda" veya "cuda:0" biçimi
        try:
            return torch.device(dev)
        except Exception:
            # geçersiz indeks vs. → cpu
            return torch.device("cpu")

    # Apple Metal
    if dev == "mps":
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device("mps")
        except Exception:
            pass
        return torch.device("cpu")

    # CPU fallback
    return torch.device("cpu")
# ============================================================
# Fonksiyon 3/14 — pick_precision
# ============================================================

def pick_precision(requested: str = "auto") -> str:
    """
    Determines numeric precision for training: "bf16" | "fp16" | "fp32"

    Kurallar:
      - requested == "auto":
          * CUDA mevcutsa:
              - If hardware supports bf16 → "bf16"
              - Aksi halde → "fp16"
          * CUDA yoksa → "fp32"
      - requested ∈ {"bf16", "fp16", "fp32"} ise doğrudan döner (güvenlik kontrolüyle).
      - Other cases → "fp32"
    """
    if isinstance(requested, str):
        req = requested.lower().strip()
    else:
        req = "auto"

    if req in {"bf16", "fp16", "fp32"}:
        return req

    if req == "auto":
        if torch.cuda.is_available():
            # bf16 destek kontrolü (Ampere+ mimarileri)
            bf16_ok = False
            try:
                # PyTorch 2.0+ için:
                if hasattr(torch.cuda, "is_bf16_supported"):
                    bf16_ok = bool(torch.cuda.is_bf16_supported())
                else:
                    # mimari üzerinden tahmin: (8.x ve üstü genelde bf16 destekler)
                    major, minor = torch.cuda.get_device_capability()
                    bf16_ok = (major >= 8)
            except Exception:
                bf16_ok = False

            return "bf16" if bf16_ok else "fp16"
        else:
            return "fp32"

    # unknown request → safe default
    return "fp32"
# ============================================================
# Fonksiyon 4/14 — get_amp_dtype
# ============================================================

def get_amp_dtype(precision: str) -> Optional[torch.dtype]:
    """
    AMP (Automatic Mixed Precision) için uygun torch.dtype döndürür.
      - "bf16" → torch.bfloat16
      - "fp16" → torch.float16
      - "fp32" → None (AMP kapalı, tam hassasiyet)

    Not:
      - torch.autocast kullanırken dtype=None → tam hassasiyet
      - Model/config tarafında `precision` ile birlikte çağrılır.
    """
    p = (precision or "").lower().strip()
    if p == "bf16":
        return torch.bfloat16
    if p == "fp16":
        return torch.float16
    return None  # fp32 veya bilinmeyen → AMP kapalı
# ============================================================
# Fonksiyon 5/14 — configure_tf32
# ============================================================

def configure_tf32(enable: bool = True) -> None:
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = bool(enable)
            torch.backends.cudnn.allow_tf32 = bool(enable)
    except Exception:
        pass
# ============================================================
# Fonksiyon 6/14 — compute_ece
# ============================================================

def compute_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    bins: int = 15,
    from_logits: bool = True,
) -> float:
    with torch.no_grad():
        # Shape correction
        if logits.dim() > 1:
            # (N,2) softmax logit geldi ise pozitif sınıfı seç
            if logits.shape[-1] == 2:
                # class-1 logit - class-0 logit → sigmoid ile p1
                logits = logits[..., 1] - logits[..., 0]
            else:
                # (N,1) ise sıkıştır
                logits = logits.squeeze(-1)
        if labels.dim() > 1:
            labels = labels.view(-1)

        logits = logits.float().detach().cpu()
        labels = labels.long().detach().cpu()

        # Convert to probability
        if from_logits:
            probs = torch.sigmoid(logits)
        else:
            probs = logits.clamp(0, 1)

        # Bin 
        bins = int(max(1, bins))
        edges = torch.linspace(0.0, 1.0, steps=bins + 1)
        ece = torch.tensor(0.0)

        N = probs.numel()
        if N == 0:
            return 0.0

        for b in range(bins):
            lo, hi = edges[b].item(), edges[b + 1].item()
            # Make the last bin right-closed: to catch p==1.0
            if b < bins - 1:
                mask = (probs >= lo) & (probs < hi)
            else:
                mask = (probs >= lo) & (probs <= hi)

            if mask.any():
                conf = probs[mask].mean()
                acc = labels[mask].float().mean()  # p(y=1)
                weight = mask.float().mean()       # |B_k| / N
                ece += weight * (acc - conf).abs()

        return float(ece.item())
# ============================================================
# Fonksiyon 7/14 — compute_brier
# ============================================================

def compute_brier(
    logits: torch.Tensor,
    labels: torch.Tensor,
    from_logits: bool = True,
) -> float:
    """
    Binary Brier Score hesaplar: mean( (p - y)^2 )
    - from_logits=True ise sigmoid(logits) kullanılır.
    - Çok boyutlu logit gelirse (N,2) → class-1 olasılığına indirger.

    Returns:
        float: Brier skoru (0 → ideal)
    """
    with torch.no_grad():
        if logits.dim() > 1:
            if logits.shape[-1] == 2:
                logits = logits[..., 1] - logits[..., 0]
            else:
                logits = logits.squeeze(-1)
        if labels.dim() > 1:
            labels = labels.view(-1)

        logits = logits.float().detach().cpu()
        labels = labels.float().detach().cpu().clamp(0, 1)

        probs = torch.sigmoid(logits) if from_logits else logits.clamp(0, 1)
        brier = torch.mean((probs - labels) ** 2)
        return float(brier.item())
# ============================================================
# Fonksiyon 8/14 — f1_vulnerable
# ============================================================

def f1_vulnerable(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> float:

    with torch.no_grad():
        # Şekilleri normalize et
        if y_pred.dim() > 1:
            if y_pred.shape[-1] == 2:
                # class-1 logit - class-0 logit
                y_pred = y_pred[..., 1] - y_pred[..., 0]
            else:
                y_pred = y_pred.squeeze(-1)
        if y_true.dim() > 1:
            y_true = y_true.view(-1)

        y_true = y_true.long().detach().cpu()
        y_pred = y_pred.float().detach().cpu()

        # Olasılık
        probs = torch.sigmoid(y_pred) if from_logits else y_pred.clamp(0, 1)
        preds_bin = (probs >= float(threshold)).long()

        tp = torch.sum((preds_bin == 1) & (y_true == 1)).item()
        fp = torch.sum((preds_bin == 1) & (y_true == 0)).item()
        fn = torch.sum((preds_bin == 0) & (y_true == 1)).item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        return float(f1)
# ============================================================
# Fonksiyon 9/14 — load_config
# ============================================================

def load_config(path_or_dict: Any) -> Dict[str, Any]:
    """
    Reads YAML config file or returns dict directly.

    Args:
        path_or_dict: str (dosya yolu) veya dict.
    Returns:
        dict: Config dictionary.

    Behavior:
      - YAML during reading '!!python' tag'lerini engeller (safe_load).
      - Returns ValueError if file path is invalid.
    """
    if isinstance(path_or_dict, dict):
        return path_or_dict

    if isinstance(path_or_dict, str):
        if not os.path.exists(path_or_dict):
            raise FileNotFoundError(f"Config dosyası not found: {path_or_dict}")
        with open(path_or_dict, "r", encoding="utf-8") as f:
            try:
                cfg = yaml.safe_load(f)
            except Exception as e:
                raise ValueError(f"Config YAML parse hatası: {e}")
        if not isinstance(cfg, dict):
            raise ValueError("Config dosyası bir YAML sözlüğü içermiyor.")
        return cfg

    raise TypeError("load_config: path_or_dict dict veya str olmalıdır.")
# ============================================================
# Fonksiyon 10/14 — save_checkpoint
# ============================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    path: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.normpath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"

    checkpoint = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "torch_version": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "meta": extra_meta or {},
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # CPU'ya taşı
    cpu_ckpt = {}
    for k, v in checkpoint.items():
        if isinstance(v, dict):
            cpu_ckpt[k] = {kk: vv.cpu() if torch.is_tensor(vv) else vv for kk, vv in v.items()}
        else:
            cpu_ckpt[k] = v

    torch.save(cpu_ckpt, tmp_path)
    os.replace(tmp_path, path)
    print(f"[Checkpoint] Kaydedildi: {path}")
# ============================================================
# Fonksiyon 11/14 — load_checkpoint
# ============================================================

def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    device: torch.device,
    strict: bool = True,
) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint dosyası not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    if "model_state" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: {path}")

    # Model yükle
    model.load_state_dict(checkpoint["model_state"], strict=strict)

    # Çıktı meta
    meta_info = {
        "epoch": checkpoint.get("epoch", 0),
        "meta": checkpoint.get("meta", {}),
        "optimizer_state": checkpoint.get("optimizer_state", None),
        "scheduler_state": checkpoint.get("scheduler_state", None),
    }

    print(f"[Checkpoint] Loaded: {path} (epoch={meta_info['epoch']})")
    return meta_info
# ============================================================
# Fonksiyon 12/14 — count_parameters
# ============================================================

def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> Dict[str, int]:

    total = 0
    trainable = 0
    for p in model.parameters():
        numel = p.numel()
        total += numel
        if p.requires_grad:
            trainable += numel

    # 32-bit float parametre başına 4 byte (yaklaşık tahmin)
    size_mb = (trainable if trainable_only else total) * 4 / (1024 ** 2)
    info = {
        "trainable": trainable,
        "total": total,
        "size_mb": round(size_mb, 2),
    }
    return info
# ============================================================
# Fonksiyon 13/14 — format_time
# ============================================================

def format_time(seconds: float) -> str:
    try:
        sec = float(seconds)
    except Exception:
        return "0s"

    if sec < 0:
        sec = 0.0

    # milisaniye için kısa yol
    if sec < 1.0:
        return f"{int(sec * 1000)}ms"

    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)

    parts = []
    if d > 0:
        parts.append(f"{d}d")
    if h > 0:
        parts.append(f"{h}h")
    if m > 0:
        parts.append(f"{m}m")
    # saniyeyi her zaman ekle (0 da olabilir)
    parts.append(f"{s}s")

    return " ".join(parts)
# ============================================================
# Fonksiyon 14/14 — colorize
# ============================================================

def colorize(text: str, color: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    base = {
        "black":   30, "red":     31, "green":  32, "yellow": 33,
        "blue":    34, "magenta": 35, "cyan":   36, "white":  37,
    }
    bright_offset = 60  # 30→90, 31→91, ...

    c = (color or "").lower().strip()
    is_bright = c.startswith("bright_")
    if is_bright:
        c = c.replace("bright_", "", 1)

    code = base.get(c, None)
    if code is None:
        return text  # bilinmeyen renk → renksiz

    if is_bright:
        code += bright_offset

    start = f"\033[{code}m"
    end = "\033[0m"
    return f"{start}{text}{end}"
