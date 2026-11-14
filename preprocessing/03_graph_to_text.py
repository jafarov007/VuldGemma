#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: convert Joern GraphSON exports into the textual graph format read by VuldGemma.

For every exported function (folder <name> produced by 02_joern_export.sh), the file-level
graph <name>/<name>/_global_.json is read and written to <out_dir>/<name>.txt:
  - vertices: only those with a non-empty CODE field (VID, LABEL, NAME, TYPE, ARGIDX, CODE)
  - edges: only those carrying an EdgeProperty and without UNKNOWN endpoints
    (in the CPG, this property is attached to REACHING_DEF edges)
Lines are prefixed with "NODE " / "EDGE ", as expected by src/data.py.

Usage: python 03_graph_to_text.py <json_dir> <out_dir>
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

def gval(x: Any):
    """Unwrap Gremlin/TinkerPop '@value' wrappers."""
    if isinstance(x, dict) and "@value" in x:
        return gval(x["@value"])
    if isinstance(x, list):
        return [gval(i) for i in x]
    return x

def first_or_none(v):
    if isinstance(v, list) and v:
        return v[0]
    return v

def clean_one_line(s: Optional[str]) -> str:
    if s is None:
        return ""
    # collapse to a single line and normalise whitespace
    out = " ".join(str(s).replace("\n", " ").replace("\r", " ").split())
    # treat placeholder values such as "<empty>" as empty
    if out.strip().lower() in {"<empty>", "null", "none"}:
        return ""
    return out

def q(s: Optional[str]) -> str:
    """Quote a string safely using JSON escaping."""
    return json.dumps("" if s is None else str(s))

def _first_non_empty(seq):
    for it in seq:
        v = prop_value_flexible(it)  # recursive
        if v is not None and str(v).strip() != "":
            return v
    return None

def prop_value_flexible(node: Any) -> Optional[str]:
    """
    Tolerant value extraction for g:VertexProperty / g:Property variants
    (wrapped values, {"key":..., "value":...} dicts, plain scalars, lists).
    """
    if node is None:
        return None

    # raw list: first non-empty item
    if isinstance(node, list):
        return _first_non_empty(node)

    # strip wrappers
    unwrapped = gval(node)

    # the unwrapped value may still be a list (e.g., g:List)
    if isinstance(unwrapped, list):
        # first non-empty item of the inner list
        for it in unwrapped:
            v = prop_value_flexible(it)  # recursive
            if v is not None and str(v).strip() != "":
                return v
        return None

    # scalar: convert to string
    if isinstance(unwrapped, (str, int, float, bool)):
        return str(unwrapped)

    # dict: try the "value" key
    if isinstance(unwrapped, dict):
        if "value" in unwrapped:
            v = gval(unwrapped.get("value"))
            if isinstance(v, list):
                # first non-empty value in the list
                for it in v:
                    vv = prop_value_flexible(it)
                    if vv is not None and str(vv).strip() != "":
                        return vv
                return None
            return "" if v is None else str(v)
        # unknown structure: no value
        return None

    return None

# ---------- extraction ----------

def extract_vertex(v: Dict[str, Any]) -> Dict[str, Any]:
    vid = gval(v.get("id"))
    label = v.get("label") or ""

    props = v.get("properties", {}) or {}

    def p(name: str) -> Optional[str]:
        node = props.get(name)
        return prop_value_flexible(node)

    code = clean_one_line(p("CODE"))
    name = p("NAME") or ""
    tfn  = p("TYPE_FULL_NAME") or ""
    argi = p("ARGUMENT_INDEX")

    return {
        "id": vid,
        "label": label,
        "CODE": code,
        "NAME": name,
        "TYPE_FULL_NAME": tfn,
        "ARGUMENT_INDEX": argi,
    }

def extract_edge(e: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # drop edges with UNKNOWN endpoints
    in_lbl = e.get("inVLabel") or ""
    out_lbl = e.get("outVLabel") or ""
    if in_lbl == "UNKNOWN" or out_lbl == "UNKNOWN":
        return None

    # keep only edges with a non-empty EdgeProperty
    props = e.get("properties", {}) or {}
    ep = props.get("EdgeProperty")
    ep_val = prop_value_flexible(ep)

    # skip empty values
    if ep_val is None or str(ep_val).strip() == "":
        return None

    return {
        "id": gval(e.get("id")),
        "inV": gval(e.get("inV")),
        "outV": gval(e.get("outV")),
        "inVLabel": in_lbl,
        "outVLabel": out_lbl,
        "label": e.get("label") or "",
        "EdgeProperty": str(ep_val),
    }

def load_graph_lists(json_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Supports the following layouts:
    1) {"@type":"tinker:graph","@value":{"edges":[...],"vertices":[...]}}
    2) {"result":{"data":[ ... g:Vertex / g:Edge ... ]}}
    3) {"data":[ ... g:Vertex / g:Edge ... ]}
    4) [ ... g:Vertex / g:Edge ... ]
    Returns: (vertices_list, edges_list)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) TinkerGraph wrapper
    if isinstance(data, dict) and data.get("@type") == "tinker:graph":
        val = gval(data.get("@value", {})) or {}
        edges = val.get("edges", []) or []
        vertices = val.get("vertices", []) or []
        return (vertices, edges)

    # 2/3/4) list-based variants
    if isinstance(data, dict) and "result" in data and isinstance(data["result"], dict) and "data" in data["result"]:
        payload = data["result"]["data"]
    elif isinstance(data, dict) and "data" in data:
        payload = data["data"]
    else:
        payload = data

    if not isinstance(payload, list):
        raise ValueError(f"Unexpected JSON layout: {json_path}")

    vertices, edges = [], []
    for item in payload:
        t = item.get("@type")
        if t == "g:Vertex":
            vertices.append(item)
        elif t == "g:Edge":
            edges.append(item)
    return (vertices, edges)

# ---------- TXT writer ----------

def write_graph_txt(out_path: str, vertices: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def _safe_id(x):
        v = gval(x.get("id"))
        try:
            return int(v)
        except Exception:
            return str(v)

    vertices_sorted = sorted(vertices, key=_safe_id)
    edges_sorted = sorted(edges, key=_safe_id)

    with open(out_path, "w", encoding="utf-8") as out:
        out.write("<GRAPH> type=CPG version=1\n")
        out.write(f"<SUMMARY> vertices={len(vertices_sorted)} edges_with_property={len(edges_sorted)} </SUMMARY>\n\n")

        out.write("### VERTICES ###\n")
        out.write("# VID LABEL NAME TYPE ARGIDX CODE\n")
        for v in vertices_sorted:
            out.write(
                f'NODE VID={gval(v.get("id"))} '
                f'LABEL={(v.get("label") or "")} '
                f'NAME={q(v.get("NAME") or "")} '
                f'TYPE={q(v.get("TYPE_FULL_NAME") or "")} '
                f'ARGIDX={"" if v.get("ARGUMENT_INDEX") in (None, "") else v.get("ARGUMENT_INDEX")} '
                f'CODE={q(v.get("CODE") or "")}\n'
            )

        out.write("\n### EDGES (EdgeProperty not empty, no UNKNOWN endpoints) ###\n")
        out.write("# EID SRC(inV) DST(outV) ETYPE VAR inVLabel outVLabel\n")
        for e in edges_sorted:
            out.write(
                f'EDGE EID={gval(e.get("id"))} '
                f'SRC={gval(e.get("inV"))} DST={gval(e.get("outV"))} '
                f'ETYPE={(e.get("label") or "")} '
                f'VAR={q(e.get("EdgeProperty") or "")} '
                f'inVLabel={(e.get("inVLabel") or "")} outVLabel={(e.get("outVLabel") or "")}\n'
            )


# ---------- main ----------

def find_global_json(sample_dir: str, name: str) -> Optional[str]:
    """File-level graph written by joern-export: <name>/<name>/_global_.json (fallback: <name>/_global_.json)."""
    for cand in (os.path.join(sample_dir, name, "_global_.json"), os.path.join(sample_dir, "_global_.json")):
        if os.path.isfile(cand):
            return cand
    return None


def convert_sample(json_path: str, out_path: str) -> Tuple[int, int]:
    vlist, elist = load_graph_lists(json_path)
    vertices = []
    for v in vlist:
        if v.get("@type") == "g:Vertex":
            vx = extract_vertex(v)
            if (vx.get("CODE") or "").strip() != "":
                vertices.append(vx)
    edges = []
    for e in elist:
        if e.get("@type") == "g:Edge":
            ed = extract_edge(e)
            if ed is not None:
                edges.append(ed)
    write_graph_txt(out_path, vertices, edges)
    return len(vertices), len(edges)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_dir", help="output directory of 02_joern_export.sh")
    ap.add_argument("out_dir", help="directory for the textual graphs (<name>.txt)")
    args = ap.parse_args()

    if not os.path.isdir(args.json_dir):
        sys.exit(f"[ERROR] input directory not found: {args.json_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    ok = err = 0
    for name in sorted(d for d in os.listdir(args.json_dir) if os.path.isdir(os.path.join(args.json_dir, d))):
        json_path = find_global_json(os.path.join(args.json_dir, name), name)
        if json_path is None:
            print(f"[SKIP] {name}: _global_.json not found")
            continue
        try:
            nv, ne = convert_sample(json_path, os.path.join(args.out_dir, f"{name}.txt"))
            print(f"[OK] {name} (V:{nv} / E:{ne})")
            ok += 1
        except Exception as ex:
            print(f"[ERROR] {name}: {ex}", file=sys.stderr)
            err += 1
    print(f"\nDone. Converted: {ok}, failed: {err}")


if __name__ == "__main__":
    main()
