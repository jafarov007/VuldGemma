import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

@dataclass
class SamplePair:
    code_path: str
    graph_path: str
    label: int

class CodeGraphDataset(Dataset):
    """
    Dataset that loads CODE and GRAPH for Gemma.
    Automatically compresses CPG graph files and pre-caches them into a binary .pt file
    for lightning-fast I/O during training.
    """
    def __init__(
        self,
        paths_cfg: Dict,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        code_max: int = 768,
        graph_max: int = 512,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.code_max = code_max
        self.graph_max = graph_max
        self.split = split
        
        # Load sample pairs
        self.pairs = self._discover_split_pairs(paths_cfg, split)
        if not self.pairs:
            raise ValueError(f"No samples found for split: {split}")
            
        # ── Binary Caching System ──
        # Processes dataset once to binary, then loads directly into RAM.
        cache_name = f"cached_{split}_c{code_max}_g{graph_max}.pt"
        self.cache_path = os.path.join(paths_cfg["root"], cache_name)
        
        if os.path.exists(self.cache_path):
            print(f"  [Veri] Fast binary cache found: {self.cache_path}")
            print("  [Veri] Loading directly into RAM (Takes a few seconds)...")
            self.samples = torch.load(self.cache_path)
        else:
            print(f"  [Veri] Cache not found. {len(self.pairs)} files are being processed and converted to binary...")
            print("  [Veri] This process will be done only once, please wait (might take 10-15 mins depending on IO):")
            self.samples = self._build_and_cache_dataset()
            print(f"  [Veri] Successfully saved as binary: {self.cache_path}")

    def _build_and_cache_dataset(self) -> List[Dict[str, Any]]:
        samples = []
        for idx, pair in enumerate(tqdm(self.pairs, desc=f"Processing {self.split}")):
            # Load code
            try:
                with open(pair.code_path, "r", encoding="utf-8", errors="replace") as f:
                    code_raw = f.read().strip()
            except Exception:
                code_raw = ""
                
            code_txt = f"### CODE ###\n{code_raw}"
            
            # Load and compress graph
            compressed_path = pair.graph_path + ".compressed"
            if os.path.exists(compressed_path):
                try:
                    with open(compressed_path, "r", encoding="utf-8", errors="replace") as f:
                        compressed_graph = f.read().strip()
                except Exception:
                    compressed_graph = ""
            else:
                try:
                    with open(pair.graph_path, "r", encoding="utf-8", errors="replace") as f:
                        graph_raw = f.read().strip()
                    compressed_graph = self._compress_graph(graph_raw)
                    # Save compressed text to disk for future use
                    with open(compressed_path, "w", encoding="utf-8") as f:
                        f.write(compressed_graph)
                except Exception:
                    compressed_graph = ""
                
            graph_txt = f"### GRAPH ###\n{compressed_graph}"

            # Tokenize code
            code_enc = self.tokenizer(
                code_txt,
                truncation=True,
                max_length=self.code_max,
                padding="max_length",
                return_tensors="pt",
            )
            
            # Tokenize graph
            graph_enc = self.tokenizer(
                graph_txt,
                truncation=True,
                max_length=self.graph_max,
                padding="max_length",
                return_tensors="pt",
            )

            samples.append({
                "code_input_ids": code_enc["input_ids"].squeeze(0),
                "code_attention_mask": code_enc["attention_mask"].squeeze(0),
                "graph_input_ids": graph_enc["input_ids"].squeeze(0),
                "graph_attention_mask": graph_enc["attention_mask"].squeeze(0),
                "label": torch.tensor(pair.label, dtype=torch.float32),
                "index": idx
            })
            
        # Save binary file
        torch.save(samples, self.cache_path)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Cache yapısı sayesinde DataLoader artık hiçbir dosyaya dokunmaz, direkt RAM'den okur!
        return self.samples[idx]

    def _compress_graph(self, graph_raw: str) -> str:
        """
        Compression Algorithm V3 (Final - MAX_EDGES=90):
        1. Keeps only 'CFG', 'CDG', 'DDG', 'REACHING_DEF' edges.
        2. Takes only the first 90 raw edges (to fit the LLM).
        3. Keeps only nodes connected to these 90 edges.
        4. Removes unnecessary nodes (LITERAL/UNKNOWN).
        """
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
                        
        # Take only the first 90 critical edges to prevent truncation
        MAX_EDGES = 90
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

    def _discover_split_pairs(self, paths_cfg: Dict, split: str) -> List[SamplePair]:
        """Automatically scans directories to pair .c files with .c.txt graph files."""
        root = paths_cfg["root"]
        code_cfg = paths_cfg["code"]
        graph_cfg = paths_cfg["graph"]
        
        pairs = []
        
        for category, label in [("vuln", 1), ("safe", 0)]:
            code_dir = os.path.join(root, code_cfg[category][split])
            graph_dir = os.path.join(root, graph_cfg[category][split])
            
            if not os.path.isdir(code_dir):
                print(f"Warning: Code dir not found: {code_dir}")
                continue
                
            for filename in os.listdir(code_dir):
                if filename.endswith(".c") or filename.endswith(".cpp"):
                    code_path = os.path.join(code_dir, filename)
                    graph_path = os.path.join(graph_dir, filename + ".txt")
                    
                    if os.path.exists(graph_path):
                        pairs.append(SamplePair(code_path, graph_path, label))
                        
        return pairs

class PadCollator:
    """Simple collator since padding is handled in dataset."""
    def __call__(self, batch):
        return {
            "code_input_ids": torch.stack([x["code_input_ids"] for x in batch]),
            "code_attention_mask": torch.stack([x["code_attention_mask"] for x in batch]),
            "graph_input_ids": torch.stack([x["graph_input_ids"] for x in batch]),
            "graph_attention_mask": torch.stack([x["graph_attention_mask"] for x in batch]),
            "labels": torch.stack([x["label"] for x in batch]).unsqueeze(-1),
            "indices": torch.tensor([x["index"] for x in batch], dtype=torch.long)
        }
