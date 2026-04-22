# KITScenes

Exploration and analysis of the [KITScenes-LongTail](https://huggingface.co/datasets/KIT-MRT/KITScenes-LongTail) dataset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Downloading the dataset

The dataset is gated. Request access at https://huggingface.co/datasets/KIT-MRT/KITScenes-LongTail, then authenticate:

```bash
huggingface-cli login
```

Download a split:

```bash
# Train split (~245 MB)
python scripts/download_data.py --splits train

# Test split (~34 GB)
python scripts/download_data.py --splits test
```

Data is saved under `data/`. That directory is git-ignored (only `data/.gitkeep` is tracked).

## Structure

```
kitscenes/
├── data/          # downloaded dataset files (git-ignored)
├── scripts/
│   └── download_data.py
├── src/
│   └── __init__.py
├── requirements.txt
└── README.md
```
