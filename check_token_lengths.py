import os
import random
import yaml
import torch
from transformers import AutoTokenizer
import re

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def compress_graph_v1(graph_raw: str) -> str:
    lines = graph_raw.splitlines()
    vid_to_id = {}
    next_id = 1
    node_lines = []
    edge_lines = []
    node_pattern = re.compile(r"NODE VID=(\d+) LABEL=(\w+).*?CODE=\"(.*?)\"")
    edge_pattern = re.compile(r"EDGE.*?SRC=(\d+) DST=(\d+) ETYPE=(\w+)")
    
    for line in lines:
        if line.startswith("NODE"):
            match = node_pattern.search(line)
            if match:
                vid, label, code = match.groups()
                if label in ["LITERAL", "UNKNOWN"]: continue
                if vid not in vid_to_id:
                    vid_to_id[vid] = f"N{next_id}"
                    next_id += 1
                nid = vid_to_id[vid]
                code_short = code[:100]
                node_lines.append(f"{nid}:[{label}] {code_short}")
        elif line.startswith("EDGE"):
            match = edge_pattern.search(line)
            if match:
                src, dst, etype = match.groups()
                if src in vid_to_id and dst in vid_to_id:
                    edge_lines.append(f"{vid_to_id[src]}-{etype}->{vid_to_id[dst]}")

    compressed = []
    if node_lines: compressed.extend(["Nodes:"] + node_lines)
    if edge_lines: compressed.extend(["Edges:"] + edge_lines)
    return "\n".join(compressed)

def compress_graph_v2(graph_raw: str) -> str:
    lines = graph_raw.splitlines()
    vid_to_id = {}
    next_id = 1
    node_lines = []
    edge_lines = []
    node_pattern = re.compile(r"NODE VID=(\d+) LABEL=(\w+).*?CODE=\"(.*?)\"")
    edge_pattern = re.compile(r"EDGE.*?SRC=(\d+) DST=(\d+) ETYPE=(\w+)")
    
    allowed_edges = {"CFG", "CDG", "DDG", "REACHING_DEF"}
    
    used_vids = set()
    raw_edges = []
    for line in lines:
        if line.startswith("EDGE"):
            match = edge_pattern.search(line)
            if match:
                src, dst, etype = match.groups()
                if etype in allowed_edges:
                    used_vids.add(src)
                    used_vids.add(dst)
                    raw_edges.append((src, dst, etype))
                    
    for line in lines:
        if line.startswith("NODE"):
            match = node_pattern.search(line)
            if match:
                vid, label, code = match.groups()
                if vid not in used_vids: continue
                if label in ["LITERAL", "UNKNOWN"]: continue
                if vid not in vid_to_id:
                    vid_to_id[vid] = f"N{next_id}"
                    next_id += 1
                nid = vid_to_id[vid]
                code_short = code[:50]
                node_lines.append(f"{nid}:[{label}] {code_short}")
                
    for src, dst, etype in raw_edges:
        if src in vid_to_id and dst in vid_to_id:
            edge_lines.append(f"{vid_to_id[src]}-{etype}->{vid_to_id[dst]}")

    compressed = []
    if node_lines: compressed.extend(["Nodes:"] + node_lines)
    if edge_lines: compressed.extend(["Edges:"] + edge_lines)
    return "\n".join(compressed)

def compress_graph_v3(graph_raw: str) -> str:
    lines = graph_raw.splitlines()
    vid_to_id = {}
    next_id = 1
    node_lines = []
    edge_lines = []
    node_pattern = re.compile(r"NODE VID=(\d+) LABEL=(\w+).*?CODE=\"(.*?)\"")
    edge_pattern = re.compile(r"EDGE.*?SRC=(\d+) DST=(\d+) ETYPE=(\w+)")
    
    allowed_edges = {"CFG", "CDG", "DDG", "REACHING_DEF"}
    
    raw_edges = []
    for line in lines:
        if line.startswith("EDGE"):
            match = edge_pattern.search(line)
            if match:
                src, dst, etype = match.groups()
                if etype in allowed_edges:
                    raw_edges.append((src, dst, etype))
                    
    MAX_EDGES = 120
    raw_edges = raw_edges[:MAX_EDGES]
    
    # Determine nodes only from the remaining edges
    used_vids = set()
    for src, dst, etype in raw_edges:
        used_vids.add(src)
        used_vids.add(dst)
                    
    for line in lines:
        if line.startswith("NODE"):
            match = node_pattern.search(line)
            if match:
                vid, label, code = match.groups()
                if vid not in used_vids: continue
                if label in ["LITERAL", "UNKNOWN"]: continue
                if vid not in vid_to_id:
                    vid_to_id[vid] = f"N{next_id}"
                    next_id += 1
                nid = vid_to_id[vid]
                code_short = code[:40]
                node_lines.append(f"{nid}:[{label}] {code_short}")
                
    for src, dst, etype in raw_edges:
        if src in vid_to_id and dst in vid_to_id:
            edge_lines.append(f"{vid_to_id[src]}-{etype}->{vid_to_id[dst]}")

    compressed = []
    if node_lines: compressed.extend(["Nodes:"] + node_lines)
    if edge_lines: compressed.extend(["Edges:"] + edge_lines)
    return "\n".join(compressed)

def main():
    cfg = load_config()
    model_name = cfg["model"]["base"]
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    root = cfg["paths"]["root"]
    code_dir = os.path.join(root, cfg["paths"]["code"]["vuln"]["train"])
    graph_dir = os.path.join(root, cfg["paths"]["graph"]["vuln"]["train"])
    
    all_files = [f for f in os.listdir(code_dir) if f.endswith(".c") or f.endswith(".cpp")]
    
    print("\n" + "="*70)
    print(" TOKEN LENGTH CALCULATION (OLD vs V1 vs V2 vs V3 Final)")
    print("="*70)

    for deneme in range(1, 4):
        random_files = random.sample(all_files, 10)
        
        kod_uz = []
        eski_grafik_uz = []
        v1_grafik_uz = []
        v2_grafik_uz = []
        v3_grafik_uz = []
        
        for file in random_files:
            code_path = os.path.join(code_dir, file)
            graph_path = os.path.join(graph_dir, file + ".txt")
            
            with open(code_path, "r", encoding="utf-8", errors="replace") as f:
                code_text = f.read().strip()
            kod_uz.append(len(tokenizer(code_text)["input_ids"]))
            
            if os.path.exists(graph_path):
                with open(graph_path, "r", encoding="utf-8", errors="replace") as f:
                    graph_raw = f.read().strip()
                
                eski_grafik_uz.append(len(tokenizer(graph_raw)["input_ids"]))
                # V1
                v1_grafik_uz.append(len(tokenizer(compress_graph_v1(graph_raw))["input_ids"]))
                # V2
                v2_grafik_uz.append(len(tokenizer(compress_graph_v2(graph_raw))["input_ids"]))
                # V3 (FINAL MAX_EDGES=120)
                v3_grafik_uz.append(len(tokenizer(compress_graph_v3(graph_raw))["input_ids"]))

        ort_kod = sum(kod_uz) / 10
        ort_eski = sum(eski_grafik_uz) / 10 if eski_grafik_uz else 0
        ort_v1 = sum(v1_grafik_uz) / 10 if v1_grafik_uz else 0
        ort_v2 = sum(v2_grafik_uz) / 10 if v2_grafik_uz else 0
        ort_v3 = sum(v3_grafik_uz) / 10 if v3_grafik_uz else 0
        
        print(f"\n--- TRY {deneme} (Random 10 FILEs) ---")
        print(f"CODE        : {ort_kod:.0f} token")
        print(f"old Graph   : {ort_eski:.0f} token")
        print(f"V1   : {ort_v1:.0f} token")
        print(f"V2    : {ort_v2:.0f} token")
        print(f"V3  : {ort_v3:.0f} token ")

if __name__ == "__main__":
    main()
