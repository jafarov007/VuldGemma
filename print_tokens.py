import torch
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-1b-it")
graph_text = "### GRAPH ###\nNODE VID=184683593732 LABEL=UNKNOWN NAME=\"\" TYPE=\"<empty>\" ARGIDX=8 CODE=\"__exctype ( _toupper )\""
tokens = tokenizer.encode(graph_text)
print("TOKENS:", tokenizer.convert_ids_to_tokens(tokens))
