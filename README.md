# vaa-analysis

Dataset and analysis code accompanying the paper:
"Political Bias in LLMs: A Cross-National Audit Using European Voting Advice Applications".

## Run the complete analysis

The script expects a data directory that contains:

- `outputs_Smartwielen/`
- `outputs_StemWijzer/`
- `outputs_Wahl-O-Mat/`
- `outputs_Wahlrechner Tschechien/`

You can provide the path in either of these ways:

### Option 1: CLI argument

```bash
python3 complete_analysis_reproducible.py --data-dir "/path/to/Individual Statements"
```

### Option 2: Environment variable

```bash
export VAA_DATA_DIR="/path/to/Individual Statements"
python3 complete_analysis_reproducible.py
```

If neither is provided, the script will also try:

- `Individual Statements/` next to `complete_analysis_reproducible.py`
- `Individual Statements/` in the current working directory

## Repository
This repository was created with the help of Cursor to provide script readabiltiy and readme text
