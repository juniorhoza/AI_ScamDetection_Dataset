# AI Scam Detection

A pipeline for building scam/phishing message classifiers — from raw web data to fine-tuned LLMs.

## Overview

This project covers the full ML lifecycle for binary text classification (scam vs. legitimate):

1. **Data extraction** — scrape candidate messages from CommonCrawl archives
2. **Data cleaning** — filter, deduplicate, and label the raw text
3. **Analysis** — explore linguistic patterns in scam content
4. **Fine-tuning** — train DistilBERT and Qwen models with LoRA adapters

## Repository Structure

```
├── dataset/
│   └── combined_scam_dataset.csv   # ~50,000 labeled messages
└── code/
    ├── Data Extraction/            # CommonCrawl scraping with checkpoint support
    ├── Data Cleaning/              # Cleaning and deduplication pipeline
    ├── scam analysis/              # Linguistic pattern analysis
    └── Model Finetune/
        ├── distilbert__finetune_model.ipynb
        └── qwen__finetune_model.ipynb
```

## Dataset

`combined_scam_dataset.csv` contains ~50,000 text samples with classifications from two models:

| Column | Description |
|---|---|
| `text` | Raw message content |
| `qwen_classification` | Qwen label (`scam` / `not scam`) |
| `qwen_confidence` | Qwen confidence score (0–1) |
| `distil_classification` | DistilBERT label (`scam` / `not scam`) |
| `distil_confidence` | DistilBERT confidence score (0–1) |

**Source:** CommonCrawl web archives  
**Task:** Binary classification — scam vs. legitimate

## Models

Both models are fine-tuned using **LoRA** (Low-Rank Adaptation) for parameter-efficient training:

- **DistilBERT** — lightweight transformer, fast inference
- **Qwen** — larger generative model for higher accuracy

All notebooks are designed to run in Google Colab with Drive integration.

## Quick Start

```python
import pandas as pd

df = pd.read_csv('dataset/combined_scam_dataset.csv')
print(df['qwen_classification'].value_counts())
print(df.head())
```

## License

See [LICENSE](LICENSE).
