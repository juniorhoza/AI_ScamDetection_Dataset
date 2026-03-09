# AI Scam Message Detection Dataset

A curated dataset of 38,000+ messages designed for training machine learning models to identify and classify scam, phishing, and fraudulent communications.

##  Dataset Overview
This dataset provides a robust foundation for NLP tasks involving fraud detection. The data was extracted from **CommonCrawl** and underwent a rigorous cleaning process to ensure high data quality for AI training.

- **File Format:** CSV
- **Task Type:** Binary Classification (Scam vs. Legitimate)
- **Primary Source:** CommonCrawl archives

##  Data Cleaning Statistics
To maintain high model accuracy, the raw data were filtered and deduplicated:


| Metric | Value |
| :--- | :--- |
| **Initial Total Rows** | 39,171 |
| **Rows After Cleaning** | 38,202 |
| **Rows Dropped** | 969 |
| **Retention Rate** | **97.53%** |

## 🚀 Getting Started
You can load the dataset directly using Python and Pandas:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('scam_dataset.csv')

# Quick look at the label distribution
print(df['label'].value_counts())
