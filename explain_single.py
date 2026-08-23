import os
import torch
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from transformers import AutoTokenizer

from src.utils import load_config
from src.data import CodeGraphDataset, PadCollator
from src.model import VuldGemmaModel

def parse_compressed_graph(graph_text):
    """Parses compressed graph text to extract node strings."""
    lines = graph_text.split("\n")
    mode = None
    node_texts = {}
    for line in lines:
        line = line.strip()
        if not line: continue
        if line == "Nodes:": mode = "nodes"; continue
        if line == "Edges:": mode = "edges"; continue
        
        if mode == "nodes":
            if ":" in line:
                nid, rest = line.split(":", 1)
                node_texts[nid] = line
    return node_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model_epX.pt")
    parser.add_argument("--filename", type=str, required=True, help="Example: chrome_-2943582541244387919@_17273.c")
    parser.add_argument("--chunk_size", type=int, default=30, help="Max code lines to display per image")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"], help="Dataset split to search the file in (default: test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda"))

    # Load Tokenizer with use_fast=True
    model_name = cfg.get("model", {}).get("base", "google/gemma-3-1b-it")
    print(f"Loading Fast Tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {args.split.capitalize()} Dataset...")
    dataset = CodeGraphDataset(
        paths_cfg=cfg["paths"], split=args.split, tokenizer=tokenizer,
        code_max=cfg["seq_lens"]["code_max"], graph_max=cfg["seq_lens"]["graph_max"]
    )
    
    target_idx = None
    for i, pair in enumerate(dataset.pairs):
        if args.filename in pair.code_path:
            target_idx = i
            break
            
    if target_idx is None:
        print(f"❌ ERROR: File '{args.filename}' not found in the {args.split} set!")
        return

    print(f"✅ File found! (Index: {target_idx})")
    pair = dataset.pairs[target_idx]
    
    # Read raw texts
    with open(pair.code_path, "r", encoding="utf-8", errors="replace") as f:
        code_raw = f.read().strip()
    code_txt = f"### CODE ###\n{code_raw}"
    
    compressed_path = pair.graph_path + ".compressed"
    with open(compressed_path, "r", encoding="utf-8", errors="replace") as f:
        compressed_graph = f.read().strip()
    graph_txt = f"### GRAPH ###\n{compressed_graph}"

    # Tokenize with offsets
    code_enc = tokenizer(code_txt, truncation=True, max_length=cfg["seq_lens"]["code_max"], padding="max_length", return_offsets_mapping=True, return_tensors="pt")
    graph_enc = tokenizer(graph_txt, truncation=True, max_length=cfg["seq_lens"]["graph_max"], padding="max_length", return_offsets_mapping=True, return_tensors="pt")

    code_offsets = code_enc["offset_mapping"][0].tolist()
    graph_offsets = graph_enc["offset_mapping"][0].tolist()

    # Parse Graph Nodes
    node_texts = parse_compressed_graph(compressed_graph)
    node_char_ranges = {}
    for nid, text in node_texts.items():
        start = graph_txt.find(text)
        if start != -1:
            node_char_ranges[nid] = (start, start + len(text))
            
    token_to_node = {}
    for i, (start, end) in enumerate(graph_offsets):
        if start == 0 and end == 0: continue
        for nid, (n_start, n_end) in node_char_ranges.items():
            if start >= n_start and end <= n_end:
                token_to_node[i] = nid
                break

    # Parse Code Lines
    lines = code_txt.split('\n')
    line_char_ranges = {}
    curr_offset = 0
    for line_idx, line in enumerate(lines):
        line_len = len(line)
        line_char_ranges[line_idx] = (curr_offset, curr_offset + line_len)
        curr_offset += line_len + 1
        
    token_to_line = {}
    for i, (start, end) in enumerate(code_offsets):
        if start == 0 and end == 0: continue
        for line_idx, (l_start, l_end) in line_char_ranges.items():
            if start >= l_start and end <= l_end:
                token_to_line[i] = line_idx
                break

    # Extract valid Node IDs and Line Indices to define matrix dimensions
    valid_nids = list(node_texts.keys())
    valid_lines = [l for l in range(len(lines)) if not lines[l].startswith("### CODE")]
    
    # Map back to continuous index for the aggregated matrix
    nid_to_idx = {nid: i for i, nid in enumerate(valid_nids)}
    line_to_idx = {l: i for i, l in enumerate(valid_lines)}

    # Load Model
    print(f"Loading model: {args.checkpoint}...")
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

    c_ids = code_enc["input_ids"].to(device)
    c_mask = code_enc["attention_mask"].to(device)
    g_ids = graph_enc["input_ids"].to(device)
    g_mask = graph_enc["attention_mask"].to(device)

    print("\nRunning model inference and extracting Heatmap...")
    amp_dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float16
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
        output = model(c_ids, c_mask, g_ids, g_mask, return_attention=True)
        
        logits = output["logits"]
        if logits.shape[-1] == 2:
            prob = torch.softmax(logits.float(), dim=-1)[0, 1].item()
        else:
            prob = torch.sigmoid(logits.float()).squeeze(-1)[0].item()

    true_label_str = "VULNERABLE" if pair.label == 1.0 else "SAFE"
    pred_label_str = "VULNERABLE" if prob >= 0.50 else "SAFE"

    print(f"True Label: {true_label_str}")
    print(f"Model Prediction: {pred_label_str} (Probability: {prob*100:.2f}%)")

    if "attn_c2g" not in output:
        print("❌ ERROR: Model did not return attn_c2g (Cross-Attention)!")
        return

    attn_matrix = output["attn_c2g"][0].cpu().float().numpy() # [L_c, L_g]
    
    # Aggregation
    agg_matrix = np.zeros((len(valid_lines), len(valid_nids)), dtype=np.float32)

    for c_idx in range(attn_matrix.shape[0]):
        for g_idx in range(attn_matrix.shape[1]):
            val = attn_matrix[c_idx, g_idx]
            if val == 0: continue
            
            nid = token_to_node.get(g_idx)
            l_idx = token_to_line.get(c_idx)
            
            if nid in nid_to_idx and l_idx in line_to_idx:
                a_c = line_to_idx[l_idx]
                a_g = nid_to_idx[nid]
                agg_matrix[a_c, a_g] += val

    print(f"Aggregated Matrix Size: {len(valid_lines)} (Lines) x {len(valid_nids)} (Nodes)")

    # Prepare labels
    y_labels = [f"{l+1}: {lines[l][:80]}" for l in valid_lines] # Truncate long lines
    x_labels = [node_texts[nid].split(":", 1)[1] if ":" in node_texts[nid] else node_texts[nid] for nid in valid_nids]

    chunk_size = args.chunk_size
    num_chunks = (len(valid_lines) + chunk_size - 1) // chunk_size
    
    clean_filename = args.filename.replace(".c.txt", "").replace(".txt", "").replace(".c", "").replace(".cpp", "")
    
    for c_idx in range(num_chunks):
        start_idx = c_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(valid_lines))
        
        chunk_y_labels = y_labels[start_idx:end_idx]
        chunk_attn = agg_matrix[start_idx:end_idx, :]
        
        # Filter Graph nodes to top 25 most important for this chunk
        g_importance = chunk_attn.sum(axis=0)
        top_g_indices = g_importance.argsort()[::-1][:25]
        top_g_indices.sort() # Preserve node sequence order
        
        chunk_x_labels = [x_labels[i] for i in top_g_indices]
        chunk_attn_filtered = chunk_attn[:, top_g_indices]
        
        plt.figure(figsize=(14, max(8, len(chunk_y_labels) * 0.35)))
        sns.heatmap(
            chunk_attn_filtered, 
            cmap="Blues", 
            cbar=True, 
            xticklabels=chunk_x_labels, 
            yticklabels=chunk_y_labels,
            linewidths=0.5, 
            linecolor='lightgray'
        )
        
        plt.title(
            f"Aggregated Cross-Attention Heatmap (Part {c_idx+1}/{num_chunks})\n"
            f"File: {args.filename}\n"
            f"True Label: {true_label_str} | Prediction: {pred_label_str} (Prob: {prob*100:.2f}%)"
        )
        plt.xlabel("CPG Nodes")
        plt.ylabel("Source Code Lines")
        
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        out_file = f"heatmap_custom_{clean_filename}_part{c_idx+1}.png"
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"PHOTO {num_chunks} Aggregated Heatmap pieces saved SUCCESSFULLY!")

if __name__ == "__main__":
    main()
