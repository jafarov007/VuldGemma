import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.utils import load_config
from src.data import CodeGraphDataset, PadCollator
from src.model import VuldGemmaModel
from src.train import Trainer

def get_top_k_indices(importance: torch.Tensor, k: int = 15):
    """
    Given a 1D importance tensor, returns the top k indices.
    """
    
    # Get top K indices
    actual_k = min(k, importance.size(0))
    top_k_indices = torch.topk(importance, actual_k).indices.tolist()
    return top_k_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model_epX.pt")
    parser.add_argument("--mask_k", type=int, default=15, help="Number of top tokens to mask")
    parser.add_argument("--samples", type=int, default=10, help="Number of TP samples to test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda"))

    model_name = cfg.get("model", {}).get("base", "google/gemma-3-1b-it")
    print(f"Loading Tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    paths_cfg = cfg["paths"]
    print("Loading Validation Dataset...")
    valid_dataset = CodeGraphDataset(
        paths_cfg=paths_cfg, split="valid", tokenizer=tokenizer,
        code_max=cfg["seq_lens"]["code_max"], graph_max=cfg["seq_lens"]["graph_max"]
    )
    
    collator = PadCollator()
    valid_loader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False,
        collate_fn=collator
    )

    print(f"Loading Model from {args.checkpoint}...")
    model = VuldGemmaModel(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    threshold = checkpoint.get("threshold", 0.5)

    trainer = Trainer(model, valid_loader, valid_loader, cfg)
    trainer.best_threshold = threshold

    print("\nPhase 1: Extracting Baseline Heatmaps...")
    amp_dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float16
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        results = trainer.extract_attention_heatmaps(valid_loader, max_samples=300)

    # Filter True Positives
    true_positives = [r for r in results if r["label"] == 1 and r["pred"] == 1]
    print(f"Found {len(true_positives)} True Positive samples. Running Occlusion on ALL of them...")
    print(f"(Will print detailed logs for the first {args.samples} samples only)")

    total_drop = 0.0
    valid_samples = 0

    print("\n" + "="*70)
    print(" OCCLUSION ANALYSIS (PERTURBATION) RESULTS ")
    print("="*70)

    for i, tp in enumerate(tqdm(true_positives, desc="Perturbing TPs")):
        prob_base = tp["prob"]
        c_ids = tp["code_tokens"].unsqueeze(0).to(device) # [1, seq_len]
        g_ids = tp["graph_tokens"].unsqueeze(0).to(device)
        
        attn_c2g = tp.get("attn_c2g") # [heads, code_len, graph_len]
        attn_c2g = tp.get("attn_c2g") # [code_len, graph_len]
        attn_g2c = tp.get("attn_g2c") # [graph_len, code_len]
        
        if attn_c2g is None or attn_g2c is None:
            continue

        # A Code token is important if many Graph tokens attend to it
        importance_code = attn_g2c.sum(dim=0) # [code_len]
        
        # A Graph token is important if many Code tokens attend to it
        importance_graph = attn_c2g.sum(dim=0) # [graph_len]

        # Get top K indices
        top_code_idx = get_top_k_indices(importance_code, k=args.mask_k)
        top_graph_idx = get_top_k_indices(importance_graph, k=args.mask_k)

        # Create masked inputs
        c_ids_masked = c_ids.clone()
        g_ids_masked = g_ids.clone()
        c_mask = torch.ones_like(c_ids_masked, dtype=torch.long)
        g_mask = torch.ones_like(g_ids_masked, dtype=torch.long)

        # MASK CODE
        for idx in top_code_idx:
            if idx < c_ids_masked.size(1):
                c_ids_masked[0, idx] = pad_id
                c_mask[0, idx] = 0 # Tell attention to ignore this completely

        # MASK GRAPH
        for idx in top_graph_idx:
            if idx < g_ids_masked.size(1):
                g_ids_masked[0, idx] = pad_id
                g_mask[0, idx] = 0

        # Phase 2: Run Inference with Masked Data
        amp_dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float16
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
            output_masked = model(c_ids_masked, c_mask, g_ids_masked, g_mask, mode="full")
            logits_masked = output_masked["logits"]
            if logits_masked.shape[-1] == 2:
                prob_masked = torch.softmax(logits_masked.float(), dim=-1)[0, 1].item()
            else:
                prob_masked = torch.sigmoid(logits_masked.float())[0].item()

        drop = prob_base - prob_masked
        total_drop += drop
        valid_samples += 1

        if i < args.samples:
            print(f"\nSample {i+1}:")
            print(f"  - Original Vulnerability Probability : {prob_base*100:.2f}%")
            print(f"  - Probability after masking Top-{args.mask_k}: {prob_masked*100:.2f}%")
            drop_str = f"{drop*100:.2f}% 📉 (Drop)" if drop > 0 else f"{abs(drop)*100:.2f}% 📈 (Increase)"
            print(f"  - Probability change: {drop_str}")
            print("-" * 50)

    if valid_samples > 0:
        avg_drop = total_drop / valid_samples
        print(f"\nSUMMARY ACROSS ALL {valid_samples} TRUE POSITIVE SAMPLES:")
        print(f"AVERAGE PROBABILITY DROP: {avg_drop*100:.2f}%")

if __name__ == "__main__":
    main()
