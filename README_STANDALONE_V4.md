# revised_reproduce_workflow_v.4 — standalone scripts with live status/logging

Generated: 2026-06-09T00:39:43

## What v.4 fixes

v.3 fixed fixed-path portability by bundling `reproduce_workflow_scripts/`.
v.4 goes one step further: **each top-level step script embeds its own source**, so the workflow no longer depends on a separate original-script folder.

## Folder layout

```text
revised_reproduce_workflow_v.4/
├── 00_check_environment.py
├── 01_fetch_g4ldb.py
├── ...
├── 27_build_reproducibility_workflow_html.py
├── environment.revised_reproduce_workflow_v4.yaml
├── workflow_manifest.json
├── workflow_manifest.csv
├── REPRODUCIBLE_WORKFLOW_STEP_BY_STEP.html
└── logs/
```

There is intentionally **no** `reproduce_workflow_scripts/` folder.

## Recommended run on a new computer

```bash
cd /path/to/revised_reproduce_workflow_v.4
micromamba env create -f environment.revised_reproduce_workflow_v4.yaml
micromamba activate nipah-g4-reproducible

python 00_check_environment.py --dry-run
python 00_check_environment.py
```

## Live status behavior

Every script prints status immediately:

- `[START]` when the run begins
- `[RUNNING]` before executing embedded source
- live stdout/stderr from the embedded source mirrored to terminal and log files
- heartbeat `[RUNNING] ... elapsed=...` every 30 seconds for quiet long steps
- `[DONE]` with final result and return code

Logs are always written under `logs/` unless `--log-dir` is supplied. Filenames use microsecond-resolution timestamps so back-to-back runs do not overwrite each other:

- `.stdout.log`
- `.stderr.log`
- `.status.json`
- `.status.md`

## Useful options

```bash
# Preview without running source
python 11_train_gnn_molecular_graph_g4_classifier.py --seed 42 --seeds 0-19 --dry-run

# Run with faster heartbeat messages
python 11_train_gnn_molecular_graph_g4_classifier.py --seed 42 --seeds 0-19 --progress-interval 10

# Forward original-script-specific args directly; unknown args are passed through
python 06_train_baseline_g4_classifier.py --seed 42 --some-original-option value

# Or use --extra-args when an option name conflicts with wrapper options
python 06_train_baseline_g4_classifier.py --seed 42 --extra-args "--help"
```

Note: unknown arguments are forwarded to the embedded source automatically. If an option conflicts with wrapper options, pass it via `--extra-args`.
