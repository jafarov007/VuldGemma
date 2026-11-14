# Data preprocessing

This folder contains the pipeline that converts C/C++ functions into the textual graph files read by `src/data.py`.

Requirements: [Joern](https://github.com/joernio/joern) **v4.0.227** and Python 3.

| Step | Script | Description |
|---|---|---|
| 1 | `01_joern_parse.sh <JOERN_CLI_DIR> <SOURCE_DIR> <CPG_OUT_DIR>` | Builds a Code Property Graph (`<name>.bin`) for every `.c`/`.cpp` function file with `joern-parse`. |
| 2 | `02_joern_export.sh <JOERN_CLI_DIR> <CPG_DIR> <JSON_OUT_DIR>` | Exports every CPG to GraphSON with `joern-export --repr cpg --format graphson`. The file-level graph is written to `<JSON_OUT_DIR>/<name>/<name>/_global_.json`. |
| 3 | `python 03_graph_to_text.py <JSON_OUT_DIR> <GRAPH_TXT_DIR>` | Writes `<name>.txt` with the vertices that have source code (`NODE` lines) and the edges that carry an edge property and have no `UNKNOWN` endpoint (`EDGE` lines). In the CPG, this property is attached to `REACHING_DEF` (data-dependence) edges. |

Step 3 reads the file-level graph either from Joern's nested layout (`<name>/<name>/_global_.json`) or from a flattened layout (`<name>/_global_.json`); flattening the directories or removing the `<includes>` folders and per-method JSON files does not change the result.

Steps 1–3 are run separately for vulnerable and non-vulnerable functions. The resulting files are placed in the directory layout expected by `config.yaml` (`vulnerables_codes`, `vulnerables_graph`, `non_vulnerables_codes`, `non_vulnerables_graph`, each with `_valid` and `_test` variants). On first use, `src/data.py` further compresses each graph (first 90 edges and their incident nodes, `LITERAL`/`UNKNOWN` nodes removed, node code truncated to 40 characters).

## Data splits

`splits/<dataset>/metadata_{train,valid,test}.json` list the exact code/graph file pairs and labels of each split used in the paper (80/10/10, class proportions preserved):

| Dataset | Train (vuln / safe) | Validation (vuln / safe) | Test (vuln / safe) |
|---|---|---|---|
| Devign | 21,855 (9,968 / 11,887) | 2,731 (1,246 / 1,485) | 2,732 (1,246 / 1,486) |
| ReVeal | 18,187 (1,792 / 16,395) | 2,273 (224 / 2,049) | 2,273 (224 / 2,049) |
