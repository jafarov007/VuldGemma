# VulDGemma — Training & Evaluation Codebase

VulDGemma is an explainable, multi-modal vulnerability detection system that combines **source code** and **Code Property Graph (CPG)** representations using Google's Gemma LLM backbone with LoRA fine-tuning and bidirectional co-attention.

The model processes two input streams simultaneously:
- **Code Stream**: Raw source code (C/C++) tokenized and encoded through Gemma.
- **Graph Stream**: Compressed CPG (Control Flow, Data Dependency, Control Dependency graphs) tokenized and encoded through the same shared Gemma encoder.

These two streams interact via a **Bidirectional Co-Attention** mechanism, allowing the model to learn cross-modal relationships between code tokens and graph nodes, which is critical for accurate vulnerability detection.

---

## Table of Contents

- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Data Preparation](#data-preparation)
- [Training](#training)
  - [Standard Training](#standard-training)
  - [Ablation Mode Training](#ablation-mode-training)
  - [Component Ablation Training](#component-ablation-training)
- [Testing & Evaluation](#testing--evaluation)
- [Automated Experiments](#automated-experiments)
- [Explainability & Visualization](#explainability--visualization)
- [Utility Scripts](#utility-scripts)
- [File Reference](#file-reference)

---

## Requirements

**Python 3.10+** and a CUDA-capable GPU with at least **30-40 GB VRAM** (e.g., A100, A6000) are required. The model uses gradient checkpointing and LoRA to fit within this budget.

Install all dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

| Package | Minimum Version | Purpose |
|---|---|---|
| torch | >= 2.0.0 | Deep learning framework |
| transformers | >= 4.45.0 | Gemma model loading & tokenizer |
| peft | >= 0.10.0 | LoRA (Parameter-Efficient Fine-Tuning) |
| pyyaml | >= 6.0 | Configuration file parsing |
| numpy | >= 1.24.0 | Numerical operations |
| tensorboard | >= 2.14.0 | Training visualization |
| scikit-learn | >= 1.3.0 | Evaluation metrics (ROC-AUC, PR-AUC, F-beta) |
| psutil | >= 5.9.0 | System resource monitoring |
| accelerate | >= 0.25.0 | HuggingFace accelerator utilities |
| safetensors | >= 0.4.0 | Safe model weight serialization |
| tqdm | >= 4.66.0 | Progress bars |
| torchao | >= 0.16.0 | PyTorch AO optimizations |

**Optional (for faster attention):**

```bash
export TORCH_CUDA_ARCH_LIST="8.0"
export FLASH_ATTN_BUILD_REDUCED_HDIM="1"
export FLASH_ATTN_HDIMS="128"
pip install flash-attn --no-build-isolation
```

---

## Project Structure

```
egitimkodlari/
├── config.yaml              # All hyperparameters and paths
├── requirements.txt         # Python dependencies
├── run_train.py             # Training entry point
├── run_test.py              # Test evaluation entry point
├── run_experiments.py       # Automated multi-seed & ablation experiments
├── run_explainability.py    # Cross-attention heatmap extraction (batch)
├── run_perturbation.py      # Occlusion-based perturbation analysis
├── explain_single.py        # Single-file heatmap (line-to-node aggregation)
├── explain_advanced.py      # Advanced visualization (graph, bipartite, HTML overlay)
├── list_predictions.py      # List all TP & FP predictions with filenames
├── check_token_lengths.py   # Token length statistics for compression strategies
├── print_tokens.py          # Quick tokenizer debug helper
├── README.md                # This file
└── src/
    ├── __init__.py           # Package marker
    ├── model.py              # VuldGemma model architecture
    ├── data.py               # Dataset loading, CPG compression, binary caching
    ├── train.py              # Trainer class (training loop, validation, EMA)
    ├── losses.py             # Focal Loss implementation
    ├── strategies.py         # ThresholdOptimizer, ModelEMA, SeedAggregator
    └── utils.py              # Helpers (metrics, logging, checkpointing, etc.)
```

---

## Configuration

All hyperparameters are centralized in `config.yaml`. Key sections:

```yaml
# General
project: vuldgemma_clean
seed: 42
device: cuda
precision: bf16          # bfloat16 for speed + memory optimization
epochs: 8
batch_size: 32
save_dir: "./runs"
ablation_mode: "full"    # "full" | "code_only" | "graph_only"

# Data Paths
paths:
  root: "../data"
  code:
    vuln:
      train: "vulnerables_codes"
      valid: "vulnerables_codes_valid"
      test:  "vulnerables_codes_test"
    safe:
      train: "non_vulnerables_codes"
      valid: "non_vulnerables_codes_valid"
      test:  "non_vulnerables_codes_test"
  graph:
    vuln:
      train: "vulnerables_graph"
      ...
    safe:
      train: "non_vulnerables_graph"
      ...

# Sequence Lengths
seq_lens:
  code_max: 1024
  graph_max: 1536

# Model
model:
  base: "google/gemma-3-1b-it"
  dropout: 0.1
  num_classes: 1
  gradient_checkpointing: true

# LoRA
lora:
  enabled: true
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Loss
loss:
  focal_gamma: 2.0
  label_smoothing: 0.1

# Strategies
strategies:
  ema_enabled: true
  ema_decay: 0.999
  optimal_threshold: true
```

You can modify `config.yaml` directly or override specific settings via CLI flags.

---

## Data Preparation

The dataset should be organized under the `paths.root` directory (default: `../data`) as follows:

```
data/
├── vulnerables_codes/          # Vulnerable C/C++ source files (train)
├── vulnerables_codes_valid/    # Vulnerable source files (validation)
├── vulnerables_codes_test/     # Vulnerable source files (test)
├── non_vulnerables_codes/      # Safe source files (train)
├── non_vulnerables_codes_valid/
├── non_vulnerables_codes_test/
├── vulnerables_graph/          # CPG graph files (.c.txt or .cpp.txt) (train)
├── vulnerables_graph_valid/
├── vulnerables_graph_test/
├── non_vulnerables_graph/
├── non_vulnerables_graph_valid/
└── non_vulnerables_graph_test/
```

- Each code file (e.g., `example.c`) must have a corresponding graph file (e.g., `example.c.txt`) in the matching graph directory.
- On the first run, the dataset module automatically compresses raw CPG graphs into a compact format (`.compressed` files) and creates a binary cache (`.pt` file) for fast subsequent loading. The initial processing may take 10-15 minutes depending on dataset size and disk I/O.

---

## Training

### Standard Training

Train the full model (Code + Graph + Co-Attention) with all components enabled:

```bash
python run_train.py --config config.yaml
```

This will:
1. Load the Gemma tokenizer and model with LoRA adapters.
2. Build train/validation datasets (with automatic binary caching).
3. Compute dynamic class weights based on the dataset distribution.
4. Train for the configured number of epochs with Focal Loss, EMA, and Optimal Threshold.
5. Save the best model checkpoint to `./runs/best_model_epX.pt`.
6. Log metrics to TensorBoard (viewable with `tensorboard --logdir runs/`).

### Ablation Mode Training

You can train using only one of the two input modalities to measure their individual contribution:

**Code-only mode** (no graph input, graph stream is zeroed):
```bash
python run_train.py --config config.yaml --mode code_only
```

**Graph-only mode** (no code input, code stream is zeroed):
```bash
python run_train.py --config config.yaml --mode graph_only
```

**Full mode** (both code and graph — this is the default):
```bash
python run_train.py --config config.yaml --mode full
```

### Component Ablation Training

You can disable individual model components to measure their impact on performance. Use the `--without-*` flags:

**Train without Focal Loss** (uses standard BCEWithLogitsLoss instead):
```bash
python run_train.py --config config.yaml --without-focal
```

**Train without EMA** (Exponential Moving Average disabled):
```bash
python run_train.py --config config.yaml --without-ema
```



These flags can be combined:
```bash
python run_train.py --config config.yaml --without-focal --without-ema
```

---

## Testing & Evaluation

After training, evaluate the saved checkpoint on the test set:

```bash
python run_test.py --config config.yaml --checkpoint runs/best_model_ep5.pt
```

This prints comprehensive metrics:
- F1 (vulnerability class), F2, Precision, Recall, Accuracy
- ROC-AUC, PR-AUC
- Confusion matrix (TP, FP, FN, TN)
- Results at both optimal threshold and fixed 0.5 threshold
- Probability distribution analysis (mean/median per class, separation gap)
- Results are automatically saved to `runs/test_results.json`.

### Testing with Mode Override

You can evaluate a checkpoint with a specific ablation mode:

```bash
python run_test.py --config config.yaml --checkpoint runs/best_model_ep5.pt --mode code_only
python run_test.py --config config.yaml --checkpoint runs/best_model_ep5.pt --mode graph_only
```

### Testing with Custom Threshold

Force a specific decision threshold:

```bash
python run_test.py --config config.yaml --checkpoint runs/best_model_ep5.pt --threshold 0.35
```

### Test-Set Threshold Search (Analysis Only)

Search for the optimal threshold on the test set. **Warning:** this causes data leakage and should only be used for analysis purposes, not for reporting final results:

```bash
python run_test.py --config config.yaml --checkpoint runs/best_model_ep5.pt --search-test
```

---

## Automated Experiments

`run_experiments.py` automates multi-seed and ablation experiments. It trains models with different configurations, aggregates results across seeds, and prints summary tables.

### Multi-Seed Robustness Test

Run the same configuration with multiple seeds to measure variance:

```bash
python run_experiments.py --config config.yaml --seeds 42,123,456
```

### Modal Ablation (Table A)

Automatically trains and compares `code_only`, `graph_only`, and `full` modes:

```bash
python run_experiments.py --config config.yaml --seeds 42,123,456 --ablation
```

### Component Ablation (Table B)

Automatically trains and compares the effect of each component (EMA, Focal Loss, Optimal Threshold):

```bash
python run_experiments.py --config config.yaml --seeds 42,123,456 --component-ablation
```

This runs three experiments per seed:
1. Full System (+EMA, +Focal, +Optimal Threshold)
2. Without Focal Loss (uses BCE)
3. Without EMA

### Full Ablation Suite

Run both modal and component ablation together:

```bash
python run_experiments.py --config config.yaml --seeds 42,123,456 --ablation --component-ablation
```

> **Warning:** Running the full ablation suite trains many models sequentially. Depending on the dataset size and GPU, this can take a **very long time** (hours to days). Each individual experiment runs a complete training cycle. Plan accordingly and consider using a multi-GPU setup or running experiments in stages.

All results are saved as JSON files in `./runs/`.

---

## Explainability & Visualization

### Batch Heatmap Extraction (`run_explainability.py`)

Extracts cross-attention heatmaps from True Positive samples to show which code tokens attend to which graph tokens:

```bash
python run_explainability.py --config config.yaml --checkpoint runs/best_model_ep5.pt --samples 5
```

Options:
- `--samples N`: Number of True Positive samples to visualize (default: 5).
- `--split {train,valid,test}`: Dataset split to evaluate (default: test).

Outputs:
- `heatmap_sample_X_partY.png`: Cross-attention heatmap images (300 DPI).
- `explainability_results.json`: Token-level attention data.

### Single-File Heatmap (`explain_single.py`)

Generate a detailed, aggregated heatmap for a specific source file. Maps cross-attention weights from token level to **source code lines vs. CPG nodes**:

```bash
python explain_single.py --config config.yaml \
    --checkpoint runs/best_model_ep5.pt \
    --filename "chrome_-2943582541244387919@_17273.c" \
    --split test
```

Options:
- `--filename`: The source code filename to analyze (must exist in the dataset).
- `--chunk_size N`: Maximum code lines per heatmap image (default: 30).
- `--split {train,valid,test}`: Which split to search the file in (default: test).

Outputs: `heatmap_custom_<filename>_partN.png` files.

### Advanced Visualization (`explain_advanced.py`)

Generates three types of advanced visualizations for a specific file:

```bash
python explain_advanced.py --config config.yaml \
    --checkpoint runs/best_model_ep5.pt \
    --filename "chrome_-2943582541244387919@_17273.c" \
    --split test
```

Outputs:
1. **CPG Graph Visualizer** (`explain_out_<file>_graph.png`): NetworkX graph visualization with nodes colored by attention importance (blue = low, red = high).
2. **Bipartite Alignment Plot** (`explain_out_<file>_bipartite.png`): Dual-panel visualization showing which code lines connect to which CPG nodes, with edge width proportional to attention strength.
3. **HTML Code Overlay** (`explain_out_<file>_overlay.html`): Interactive HTML file showing the source code with lines highlighted based on cross-attention intensity (red = high attention from graph nodes).

### Perturbation Analysis (`run_perturbation.py`)

Validates that the model's attention is causally meaningful (not just correlative) by masking the top-attended tokens and measuring probability drop:

```bash
python run_perturbation.py --config config.yaml \
    --checkpoint runs/best_model_ep5.pt \
    --mask_k 15 \
    --samples 10
```

Options:
- `--mask_k N`: Number of top-attended tokens to mask in both code and graph streams (default: 15).
- `--samples N`: Number of True Positive samples to display detailed logs for (default: 10). All TPs are processed for the summary.

Outputs: Prints per-sample probability change and an overall average probability drop. A high average drop confirms that the model relies on the structural features highlighted in the heatmaps.

---

## Utility Scripts

### List Predictions (`list_predictions.py`)

Lists all True Positive (TP) and False Positive (FP) predictions with filenames and probabilities:

```bash
python list_predictions.py --config config.yaml \
    --checkpoint runs/best_model_ep5.pt \
    --split test \
    --output prediction_results.txt
```

Options:
- `--split {train,valid,test}`: Dataset split to evaluate (default: test).
- `--output`: Output file path (default: `prediction_results.txt`).

### Check Token Lengths (`check_token_lengths.py`)

Analyzes token lengths across different CPG compression strategies (V1, V2, V3) to help decide on `graph_max` sequence length:

```bash
python check_token_lengths.py
```

Reads `config.yaml` automatically. Samples 10 random files 3 times and reports average token counts for code and each graph compression variant.

### Print Tokens (`print_tokens.py`)

Quick debug script to see how the Gemma tokenizer splits a sample graph text:

```bash
python print_tokens.py
```

---

## File Reference

### Entry Point Scripts

| File | Description |
|---|---|
| `run_train.py` | Main training entry point. Loads config, builds datasets/model, and starts the Trainer. Supports `--mode`, `--without-ema`, `--without-focal` flags. |
| `run_test.py` | Test evaluation script. Loads a checkpoint and evaluates on the test set. Reports F1, F2, Precision, Recall, Accuracy, ROC-AUC, PR-AUC, confusion matrix, and probability distributions. Supports `--mode`, `--threshold`, `--search-test` flags. |
| `run_experiments.py` | Automated experiment runner. Handles multi-seed training, modal ablation (`--ablation`), and component ablation (`--component-ablation`). Aggregates results with mean/std and saves JSON reports. |
| `run_explainability.py` | Batch cross-attention heatmap extractor. Uses Trainer's `extract_attention_heatmaps()` to get attention weights, then generates PNG heatmaps and JSON data for True Positive samples. |
| `run_perturbation.py` | Occlusion-based perturbation test. Masks top-K attended tokens in both streams, re-runs inference, and measures probability change to validate attention causality. |
| `explain_single.py` | Single-file explainability tool. Tokenizes a specific file with offset mapping, runs inference, then aggregates token-level cross-attention to **source code lines vs. CPG nodes** for interpretable heatmaps. |
| `explain_advanced.py` | Advanced multi-visualization tool. Generates CPG graph visualizations (NetworkX), bipartite code-to-graph alignment plots, and interactive HTML code overlays with attention-based highlighting. |
| `list_predictions.py` | Prediction lister. Runs inference on an entire split and outputs all TP and FP predictions sorted by probability, along with source file paths. |
| `check_token_lengths.py` | Token length analyzer. Compares old (raw) graph tokenization vs. V1, V2, V3 compression strategies to find optimal `graph_max` settings. |
| `print_tokens.py` | Tokenizer debug helper. Prints the token decomposition of a hardcoded graph text sample. |

### Source Module (`src/`)

| File | Description |
|---|---|
| `model.py` | **VuldGemmaModel** — Full architecture implementation. Contains: `CoAttentionBlock` (bidirectional cross-attention with optional weight extraction), `AttentionPooling` (single-query attention pooling), `FusionBlock` (concatenation + residual fusion), `ClassifierHead` (GELU + LayerNorm + Linear), and the main `VuldGemmaModel` class that assembles everything with shared Gemma encoder + LoRA. Supports `code_only`, `graph_only`, and `full` forward modes. |
| `data.py` | **CodeGraphDataset** — Dataset class that discovers code/graph file pairs, compresses CPG graphs using V3 compression (keeps CFG/CDG/DDG/REACHING_DEF edges, max 90 edges, removes LITERAL/UNKNOWN nodes), and caches the tokenized dataset as a binary `.pt` file for instant loading on subsequent runs. Also contains `PadCollator` for DataLoader. |
| `train.py` | **Trainer** — Complete training loop with AdamW optimizer, cosine warmup scheduler, AMP (bf16/fp16), gradient accumulation, gradient clipping, Focal Loss or BCE, EMA weight updates, optimal threshold search on validation, TensorBoard logging, and best model checkpointing. Also implements `extract_attention_heatmaps()` for explainability. |
| `losses.py` | **FocalLoss** — Focal Loss implementation for imbalanced binary classification. Adds a `(1-pt)^gamma` scaling factor to BCEWithLogitsLoss to focus learning on hard examples. Supports `pos_weight`, `label_smoothing`, and `gamma` parameters. |
| `strategies.py` | **ThresholdOptimizer** — Searches thresholds from 0.05 to 0.95 to find the one maximizing F1. **ModelEMA** — Exponential Moving Average of model weights for smoother validation and better generalization. **SeedAggregator** — Collects metrics from multiple seed runs and computes mean ± std summaries. |
| `utils.py` | Utility collection: `AverageMeter` (running average tracker), `MetricTracker` (batch-level F1/Precision/Recall/PR-AUC), `Logger` (TensorBoard + text file logging), `Timer` (execution time measurement), `set_seed` (reproducibility), `setup_device`, `pick_precision`, `get_amp_dtype`, `configure_tf32`, `compute_ece` (Expected Calibration Error), `compute_brier` (Brier Score), `f1_vulnerable` (F1 computation from logits), `load_config` (YAML parser), `save_checkpoint` / `load_checkpoint` (model serialization), `count_parameters`, `format_time`, `colorize` (terminal color output). |

---

## Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the full model
python run_train.py --config config.yaml

# 3. Evaluate on test set
python run_test.py --config config.yaml --checkpoint runs/best_model_ep5.pt

# 4. Generate explainability heatmaps
python run_explainability.py --config config.yaml --checkpoint runs/best_model_ep5.pt

# 5. Run full ablation experiments (may take a very long time)
python run_experiments.py --config config.yaml --seeds 42,123,456 --ablation --component-ablation
```
