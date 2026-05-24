# Atrial Fibrillation (AF) Detection via DCCA Feature Fusion

This project implements the automated detection of Atrial Fibrillation from short single-lead ECG recordings based on the paper **"Automated Atrial Fibrillation Detection Based on Feature Fusion Using Discriminant Canonical Correlation Analysis"** (2021).

It resolves several structural limitations of the original paper:
1. **No zero-padding contamination**: Implements dynamic attention masking for the GRU layers.
2. **Strict split isolation**: Performs DCCA projection fitting within cross-validation folds to avoid train-test data leakage.
3. **No local dependencies**: Ready for Kaggle running with GPU acceleration.

---

## Project Structure

```
├── src/
│   ├── dataset.py         # Data download, Butterworth filtering, and PyTorch dataset Loader
│   ├── features.py        # ECG feature extraction (R-peaks, RR intervals, P-wave, PSD power bands)
│   ├── models.py          # PyTorch ResNet-1D + GRU, DCCA, and Fusion Classifier
│   ├── utils.py           # Evaluation metrics (F1-scores, accuracy, plot utils)
│   └── train.py           # Cross-validation & training pipeline
├── requirements.txt       # Python package requirements
├── test_pipeline.py       # End-to-end local test pipeline with mock data
└── kaggle_run.ipynb       # Jupyter Notebook to upload and run on Kaggle
```

---

## How to Run on Kaggle

### Option A: Via GitHub (Recommended)
1. Initialize Git in this directory:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of AF Detection project"
   ```
2. Create a repository on GitHub, link it, and push:
   ```bash
   git remote add origin <your-github-repo-url>
   git branch -M main
   git push -u origin main
   ```
3. Go to **Kaggle** -> **Create New Notebook**.
4. In the Kaggle notebook cell, clone your repository and run:
   ```bash
   !git clone <your-github-repo-url>
   %cd <your-github-repo-name>
   !pip install -r requirements.txt
   !python test_pipeline.py
   !python src/train.py --data_root ./data --folds 5 --deep_epochs 20 --classifier_epochs 40
   ```

### Option B: Upload Directly
1. Zip this directory (excluding `data/` and python cache folders).
2. Go to **Kaggle** -> **Create New Notebook**.
3. Under the **"File"** menu, select **"Upload utility script"** or click **"Add Data"** and upload the zip file.
4. Unzip and run the notebook `kaggle_run.ipynb` directly.

---

## Configuration Options

You can customize the pipeline run via command line flags in `src/train.py`:
- `--data_root`: Directory to store downloaded data (default: `./data`).
- `--folds`: Number of Stratified CV folds (default: `5`).
- `--max_len_sec`: Length in seconds to crop/pad signals (default: `30`).
- `--gru_hidden`: Number of hidden units in GRU (default: `64`).
- `--dcca_dim`: Number of components to project onto using DCCA (default: `16`).
- `--dcca_eta`: Weight parameter to prioritize intra-class vs inter-class covariance (default: `0.1`).
- `--deep_epochs`: Training epochs for ResNet-GRU (default: `15`).
- `--classifier_epochs`: Training epochs for the fusion classification network (default: `30`).
- `--batch_size`: Batch size for training (default: `64`).
