import os
import torch
import json
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.utils import load_config, MetricTracker
from src.data import CodeGraphDataset, PadCollator
from src.model import VuldGemmaModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model_epX.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--output", type=str, default="prediction_results.txt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda"))
    precision = cfg.get("precision", "bf16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    # Load Tokenizer
    model_name = cfg.get("model", {}).get("base", "google/gemma-1.1-1b-it")
    print(f"Loading Tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Dataset
    paths_cfg = cfg["paths"]
    print(f"Loading {args.split.capitalize()} Dataset...")
    dataset = CodeGraphDataset(
        paths_cfg=paths_cfg, split=args.split, tokenizer=tokenizer,
        code_max=cfg["seq_lens"]["code_max"], graph_max=cfg["seq_lens"]["graph_max"]
    )
    
    collator = PadCollator()
    loader = DataLoader(dataset, batch_size=cfg.get("batch_size", 4), shuffle=False, collate_fn=collator)

    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = VuldGemmaModel(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Running Inference on all samples...")
    results = []
    
    # We will use 0.50 as the decision threshold. 
    # (Alternatively, you can change this to the optimal threshold found during training)
    threshold = 0.50 

    tracker = MetricTracker()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            c_ids = batch["code_input_ids"].to(device)
            c_mask = batch["code_attention_mask"].to(device)
            g_ids = batch["graph_input_ids"].to(device)
            g_mask = batch["graph_attention_mask"].to(device)
            labels = batch["labels"].to(device)
            indices = batch["indices"].tolist()

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True):
                output = model(c_ids, c_mask, g_ids, g_mask)
                logits = output["logits"]

            if logits.ndim == 2 and logits.shape[-1] == 2:
                probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            else:
                probs = torch.sigmoid(logits.float()).squeeze(-1)

            tracker.update(probs, labels)

            for i in range(len(labels)):
                prob = probs[i].item()
                label = int(labels[i].item())
                
                # Fetch filename using dataset.pairs
                dataset_idx = indices[i]
                filename = dataset.pairs[dataset_idx].code_path
                
                pred = 1 if prob >= threshold else 0
                
                if label == 1 and pred == 1:
                    category = "TP" # True Positive
                elif label == 0 and pred == 1:
                    category = "FP" # False Positive
                else:
                    continue # Sadece TP ve FP isteniyor
                    
                results.append({
                    "filename": filename,
                    "prob": prob,
                    "label": label,
                    "category": category
                })

    # Group by category
    categorized = {"TP": [], "FP": []}
    for r in results:
        categorized[r["category"]].append(r)

    print(f"Writing detailed results to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("VuldGemma TP & FP List\n")
        f.write("="*80 + "\n\n")
        
        for cat in ["TP", "FP"]:
            f.write(f"[{cat}] - Total: {len(categorized[cat])}\n")
            f.write("-" * 50 + "\n")
            # Sort by probability (highest to lowest)
            sorted_cat = sorted(categorized[cat], key=lambda x: x["prob"], reverse=True)
            for item in sorted_cat:
                f.write(f"Prob: {item['prob']*100:05.2f}% | File: {item['filename']}\n")
            f.write("\n")
            
    print("✅ Done! You can check the output file for all filenames and probabilities.")

if __name__ == "__main__":
    main()
