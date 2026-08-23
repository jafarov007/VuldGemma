import os
import torch
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from transformers import AutoTokenizer
import matplotlib.colors as mcolors

from src.utils import load_config
from src.data import CodeGraphDataset, PadCollator
from src.model import VuldGemmaModel

def parse_compressed_graph(graph_text):
    """Parses compressed graph text into NetworkX graph and node strings."""
    G = nx.DiGraph()
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
                G.add_node(nid, label=rest)
                node_texts[nid] = line
        elif mode == "edges":
            if "->" in line:
                src_part, dst = line.split("->")
                src, etype = src_part.split("-", 1)
                etype = etype.strip("-")
                G.add_edge(src, dst, type=etype)
    return G, node_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda"))

    model_name = cfg.get("model", {}).get("base", "google/gemma-1.1-1b-it")
    print(f"Loading Fast Tokenizer: {model_name}")
    # MUST use fast tokenizer for offset_mapping
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Dataset...")
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
        print(f"ERROR: File '{args.filename}' not found.")
        return

    sample = dataset[target_idx]
    pair = dataset.pairs[target_idx]
    
    # We need the raw text to map offsets
    with open(pair.code_path, "r", encoding="utf-8", errors="replace") as f:
        code_raw = f.read().strip()
    code_txt = f"### CODE ###\n{code_raw}"
    
    compressed_path = pair.graph_path + ".compressed"
    with open(compressed_path, "r", encoding="utf-8", errors="replace") as f:
        compressed_graph = f.read().strip()
    graph_txt = f"### GRAPH ###\n{compressed_graph}"

    # Get offset mappings
    code_enc = tokenizer(code_txt, truncation=True, max_length=cfg["seq_lens"]["code_max"], padding="max_length", return_offsets_mapping=True, return_tensors="pt")
    graph_enc = tokenizer(graph_txt, truncation=True, max_length=cfg["seq_lens"]["graph_max"], padding="max_length", return_offsets_mapping=True, return_tensors="pt")

    code_offsets = code_enc["offset_mapping"][0].tolist()
    graph_offsets = graph_enc["offset_mapping"][0].tolist()

    # Parse Graph
    G, node_texts = parse_compressed_graph(compressed_graph)
    
    # Map chars to Nodes
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

    # Map chars to Code Lines
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

    print("Running Inference...")
    amp_dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float16
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
        output = model(c_ids, c_mask, g_ids, g_mask, return_attention=True)
    
    attn_matrix = output["attn_c2g"][0].cpu().float().numpy() # [L_c, L_g]
    
    # Aggregation
    node_importance = {nid: 0.0 for nid in G.nodes()}
    line_importance = {l: 0.0 for l in range(len(lines))}
    line_to_node_attn = {}

    for c_idx in range(attn_matrix.shape[0]):
        for g_idx in range(attn_matrix.shape[1]):
            val = attn_matrix[c_idx, g_idx]
            if val == 0: continue
            
            nid = token_to_node.get(g_idx)
            l_idx = token_to_line.get(c_idx)
            
            if nid: node_importance[nid] += val
            if l_idx is not None: line_importance[l_idx] += val
            
            if nid and l_idx is not None:
                if (l_idx, nid) not in line_to_node_attn:
                    line_to_node_attn[(l_idx, nid)] = 0.0
                line_to_node_attn[(l_idx, nid)] += val

    # Normalize importances for visualization
    max_node_imp = max(node_importance.values()) if node_importance else 1.0
    if max_node_imp == 0: max_node_imp = 1.0
    
    max_line_imp = max(line_importance.values()) if line_importance else 1.0
    if max_line_imp == 0: max_line_imp = 1.0
    
    clean_filename = args.filename.replace(".c.txt", "").replace(".txt", "").replace(".c", "").replace(".cpp", "")

    # ==========================================
    # 1. Bounded Sub-graph Visualizer
    # ==========================================
    print("Generating Graph Visualizer...")
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    node_colors = []
    for nid in G.nodes():
        norm_val = min(1.0, node_importance[nid] / max_node_imp)
        # color from lightblue to darkred
        color = plt.cm.coolwarm(norm_val)
        node_colors.append(color)
        
    labels = {nid: G.nodes[nid]['label'][:15] + "..." if len(G.nodes[nid]['label'])>15 else G.nodes[nid]['label'] for nid in G.nodes()}
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, edgecolors='black')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, alpha=0.6)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')
    
    plt.title(f"Taint Path CPG Visualizer - {args.filename}")
    plt.axis("off")
    out_graph = f"explain_out_{clean_filename}_graph.png"
    plt.savefig(out_graph, dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 2. Dual-Panel Code to Graph Bipartite
    # ==========================================
    print("Generating Bipartite Alignment Plot...")
    B = nx.DiGraph()
    
    # Filter highly attended lines and nodes
    top_lines = [l for l, imp in line_importance.items() if imp > max_line_imp * 0.1]
    top_nodes = [n for n, imp in node_importance.items() if imp > max_node_imp * 0.1]
    
    # Ensure they are visually separated in layout
    for l in top_lines:
        B.add_node(f"L{l}", bipartite=0, label=f"L{l}: {lines[l][:30]}")
    for n in top_nodes:
        B.add_node(n, bipartite=1, label=G.nodes[n]['label'][:30])
        
    max_edge_weight = max(line_to_node_attn.values()) if line_to_node_attn else 1.0
    for (l, n), weight in line_to_node_attn.items():
        if l in top_lines and n in top_nodes and weight > max_edge_weight * 0.1:
            B.add_edge(f"L{l}", n, weight=weight)
            
    if len(B.nodes) > 0:
        plt.figure(figsize=(16, max(8, len(top_lines)*0.5)))
        pos = {}
        left_nodes = [n for n, d in B.nodes(data=True) if d['bipartite']==0]
        right_nodes = [n for n, d in B.nodes(data=True) if d['bipartite']==1]
        
        # Sort left vertically by line number
        left_nodes.sort(key=lambda x: int(x[1:]))
        
        for i, node in enumerate(left_nodes): pos[node] = (0, len(left_nodes) - i)
        for i, node in enumerate(right_nodes): pos[node] = (1, len(right_nodes) - i)
        
        nx.draw_networkx_nodes(B, pos, nodelist=left_nodes, node_color='lightblue', node_size=2000, node_shape='s')
        nx.draw_networkx_nodes(B, pos, nodelist=right_nodes, node_color='lightgreen', node_size=2000, node_shape='s')
        
        edges = B.edges()
        weights = [B[u][v]['weight'] / max_edge_weight * 5 for u, v in edges]
        nx.draw_networkx_edges(B, pos, edgelist=edges, width=weights, edge_color='red', alpha=0.5)
        
        labels = {n: B.nodes[n]['label'] for n in B.nodes()}
        nx.draw_networkx_labels(B, pos, labels=labels, font_size=9)
        
        plt.title(f"Dual-Panel Code-to-Graph Alignment - {args.filename}")
        plt.axis("off")
        out_bipartite = f"explain_out_{clean_filename}_bipartite.png"
        plt.savefig(out_bipartite, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("No strong bipartite edges found.")

    # ==========================================
    # 3. Integrated Code Overlay (HTML)
    # ==========================================
    print("Generating HTML Code Overlay...")
    html_content = f"""
    <html>
    <head>
        <title>Code Overlay - {args.filename}</title>
        <style>
            body {{ font-family: monospace; background: #1e1e1e; color: #d4d4d4; padding: 20px; }}
            .line {{ padding: 2px 5px; margin: 1px 0; border-radius: 3px; display: block; }}
            .ln {{ color: #858585; margin-right: 15px; user-select: none; }}
        </style>
    </head>
    <body>
        <h2>Integrated Code Overlay: {args.filename}</h2>
        <p>Lines highlighted based on cross-attention to CPG (Red = High Attention)</p>
        <pre>
    """
    
    for l_idx, line in enumerate(lines):
        if line.startswith("### CODE"): continue
        
        imp = line_importance.get(l_idx, 0)
        norm_imp = min(1.0, imp / max_line_imp)
        
        # Color: Dark background, adding red based on attention
        # rgba(255, 0, 0, alpha)
        alpha = norm_imp * 0.8 # max 80% opacity red
        bg_color = f"rgba(255, 80, 80, {alpha:.3f})"
        
        escaped_line = line.replace("<", "&lt;").replace(">", "&gt;")
        html_content += f'<span class="line" style="background-color: {bg_color};"><span class="ln">{l_idx:4d}</span>{escaped_line}</span>\n'
        
    html_content += """
        </pre>
    </body>
    </html>
    """
    
    out_html = f"explain_out_{clean_filename}_overlay.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("All advanced visualizations generated successfully!")
    print(f"Files saved: {out_graph}, {out_bipartite}, {out_html}")

if __name__ == "__main__":
    main()
