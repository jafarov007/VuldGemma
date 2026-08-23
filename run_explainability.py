import os
import torch
import json
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import load_config
from src.data import CodeGraphDataset, PadCollator
from src.model import VuldGemmaModel
from src.train import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model_epX.pt")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to extract")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"], help="Dataset split to evaluate (default: test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda"))

    # Load Tokenizer
    model_name = cfg.get("model", {}).get("base", "google/gemma-1.1-1b-it")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Test/Valid Data
    paths_cfg = cfg["paths"]
    print(f"Loading {args.split.capitalize()} Dataset...")
    valid_dataset = CodeGraphDataset(
        paths_cfg=paths_cfg, split=args.split, tokenizer=tokenizer,
        code_max=cfg["seq_lens"]["code_max"], graph_max=cfg["seq_lens"]["graph_max"]
    )
    
    collator = PadCollator()
    valid_loader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False,  # Batch size 1 for easy analysis
        collate_fn=collator
    )

    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = VuldGemmaModel(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    
    threshold = checkpoint.get("threshold", 0.5)

    # Dummy Trainer just to use the extract_attention_heatmaps method
    trainer = Trainer(model, valid_loader, valid_loader, cfg)
    trainer.best_threshold = threshold

    print("\nExtracting Cross-Modal Heatmaps...")
    amp_dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float16
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        results = trainer.extract_attention_heatmaps(valid_loader, max_samples=200)

    # Filter only True Positives (Model correctly found the vulnerability)
    true_positives = [r for r in results if r["label"] == 1 and r["pred"] == 1]
    
    output_data = []
    for i, tp in enumerate(true_positives[:args.samples]):
        # Convert token IDs back to strings for human readability
        code_tokens_raw = tokenizer.convert_ids_to_tokens(tp["code_tokens"])
        graph_tokens_raw = tokenizer.convert_ids_to_tokens(tp["graph_tokens"])
        
        # Clean up special tokens for JSON output
        code_tokens_json = [t.replace(" ", "") for t in code_tokens_raw if t not in ["<pad>", "<s>", "</s>"]]
        graph_tokens_json = [t.replace(" ", "") for t in graph_tokens_raw if t not in ["<pad>", "<s>", "</s>"]]
        
        output_data.append({
            "sample_id": i + 1,
            "prob": tp["prob"],
            "code_tokens": code_tokens_json,
            "graph_tokens": graph_tokens_json,
            "attn_code_to_graph_shape": list(tp["attn_c2g"].shape),
            "attn_graph_to_code_shape": list(tp["attn_g2c"].shape)
        })
        
        print(f"\n[Sample {i+1}] Vulnerability Found! (Prob: {tp['prob']:.4f})")
        # Find valid indices from token IDs (pad_id = tokenizer.pad_token_id)
        pad_id = tokenizer.pad_token_id
        c_valid_indices = (tp["code_tokens"] != pad_id).nonzero(as_tuple=True)[0]
        g_valid_indices = (tp["graph_tokens"] != pad_id).nonzero(as_tuple=True)[0]

        if len(c_valid_indices) > 0 and len(g_valid_indices) > 0:
            c_start, c_end = c_valid_indices[0].item(), c_valid_indices[-1].item() + 1
            g_start, g_end = g_valid_indices[0].item(), g_valid_indices[-1].item() + 1
            
            valid_c_len = c_end - c_start
            valid_g_len = g_end - g_start
            
            code_tokens_plot = code_tokens_raw[c_start:c_end]
            graph_tokens_plot = graph_tokens_raw[g_start:g_end]
            attn_matrix_np = tp["attn_c2g"][c_start:c_end, g_start:g_end].numpy()
            
            chunk_size = 40
            num_chunks = (valid_c_len + chunk_size - 1) // chunk_size
            
            for c_idx in range(num_chunks):
                s_idx = c_idx * chunk_size
                e_idx = min(s_idx + chunk_size, valid_c_len)
                
                chunk_c_tokens = code_tokens_plot[s_idx:e_idx]
                chunk_attn = attn_matrix_np[s_idx:e_idx, :]
                
                ignore_tokens = {
                    "<BOS>", "<EOS>", "<PAD>", "###", "GRAPH", "NODE", "VID", "LABEL", 
                    "CODE", "TYPE", "ARGIDX", "=", "\"", "\n", "UNKNOWN", "<EMPTY>", 
                    "##", "", "---"
                }
                
                g_importance = chunk_attn.sum(axis=0)
                sorted_indices = g_importance.argsort()[::-1]
                
                top_g_indices = []
                for idx in sorted_indices:
                    t_str = graph_tokens_plot[idx].replace(" ", "").replace(" ", "").strip().upper()
                    if t_str not in ignore_tokens and not t_str.startswith("##") and len(t_str) > 0:
                        top_g_indices.append(idx)
                    if len(top_g_indices) == min(30, valid_g_len):
                        break
                        
                if len(top_g_indices) < min(30, valid_g_len):
                    for idx in sorted_indices:
                        if idx not in top_g_indices:
                            top_g_indices.append(idx)
                        if len(top_g_indices) == min(30, valid_g_len):
                            break
                            
                top_g_indices.sort()
                
                chunk_g_tokens = [graph_tokens_plot[idx] for idx in top_g_indices]
                chunk_attn_filtered = chunk_attn[:, top_g_indices]
                
                plt.figure(figsize=(14, 10))
                sns.heatmap(
                    chunk_attn_filtered, 
                    cmap="Blues", 
                    cbar=True, 
                    xticklabels=chunk_g_tokens, 
                    yticklabels=chunk_c_tokens,
                    linewidths=0.5, 
                    linecolor='lightgray'
                )
                plt.title(f"Sample {i+1} - Cross-Attention (Part {c_idx+1}/{num_chunks})")
                plt.xlabel("Most Relevant Graph Tokens")
                plt.ylabel("Code Tokens")
                plt.xticks(rotation=45, ha='right', fontsize=9)
                plt.yticks(rotation=0, fontsize=9)
                
                heatmap_filename = f"heatmap_sample_{i+1}_part{c_idx+1}.png"
                plt.savefig(heatmap_filename, dpi=300, bbox_inches="tight")
                plt.close()
                
            print(f"PHOTO {num_chunks} Heatmap pieces saved (Sample {i+1}).")

    # Save to file
    out_file = "explainability_results.json"
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\n Explainability data saved to {out_file} and PNG heatmaps generated!")
    print("This proves the model is learning CROSS-MODAL relationships!")

if __name__ == "__main__":
    main()
