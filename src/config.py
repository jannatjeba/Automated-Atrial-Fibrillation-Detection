import os

# ==========================================
# Dataset & Paths Configuration
# ==========================================
# File that contains the path to the dataset on local machine
DATASET_PATH_FILE = "dataset_path.txt"

# Default data root directory if the path file is not found
DEFAULT_DATA_ROOT = "./data"

# Kaggle-specific configuration
KAGGLE_INPUT_DIR = "/kaggle/input"

# ==========================================
# Training Hyperparameters
# ==========================================
FOLDS = 5
MAX_LEN_SEC = 30
GRU_HIDDEN = 128  # Increased hidden dimension (previously 64)
DCCA_DIM = 16
DCCA_ETA = 0.1
BATCH_SIZE = 64
DEEP_EPOCHS = 15
CLASSIFIER_EPOCHS = 30
SEED = 42

# Optimization parameters
DEEP_LR = 0.001
DEEP_WD = 1e-4
CLASSIFIER_LR = 0.005
CLASSIFIER_WD = 1e-3

# ==========================================
# Advanced Model & Training Improvements
# ==========================================
# Loss Function Improvement
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0

# ECG Preprocessing Improvements
DETREND_SIGNAL = True            # Remove baseline wander using scipy.signal.detrend
WAVELET_DENOISING = True          # Optional Wavelet denoising using pywt
PADDING_MODE = 'reflect'          # 'zero' or 'reflect'

# Deep Feature Extractor Improvements
RNN_TYPE = 'BiLSTM'               # 'BiLSTM' or 'BiGRU'
USE_ATTENTION_POOLING = True      # Add temporal attention pooling after RNN

# Data Augmentation Improvements
USE_DATA_AUGMENTATION = True      # Apply random noise, scaling, shift, crop
MIXUP_ALPHA = 0.2                 # Alpha parameter for Mixup training (0 to disable)

# Feature Fusion Improvements
# Options: 'DCCA', 'Concatenate' (Direct Concatenation baseline), 'Attention' (Gated Attention fusion)
FUSION_METHOD = 'DCCA'

# Training Strategy Improvements
USE_COSINE_SCHEDULER = True       # CosineAnnealingLR
USE_EARLY_STOPPING = True         # Early stopping on validation loss
EARLY_STOPPING_PATIENCE = 5
USE_GRAD_CLIPPING = True          # Clip gradients to max_norm 1.0
USE_AMP = True                    # Mixed Precision training (float16)

# Class Imbalance Handling
USE_BALANCED_SAMPLER = True       # WeightedRandomSampler for balanced batches

# CNN Architecture Improvements
USE_SE_BLOCK = True               # Add Squeeze-and-Excitation (SE) blocks to ResNet

def get_dataset_path():
    """
    Dynamically determines the dataset root directory.
    1. Checks if running on Kaggle. If so, scans /kaggle/input for the dataset.
    2. Otherwise, tries to read the path from DATASET_PATH_FILE.
    3. If the file is not found, falls back to DEFAULT_DATA_ROOT.
    """
    # 1. Check if running on Kaggle
    is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists(KAGGLE_INPUT_DIR)
    if is_kaggle:
        print("Kaggle environment detected. Scanning for dataset in /kaggle/input...")
        # Search for REFERENCE.csv in /kaggle/input
        for root, dirs, files in os.walk(KAGGLE_INPUT_DIR):
            if 'REFERENCE.csv' in files:
                # If REFERENCE.csv is inside a subdirectory named training2017,
                # we want the parent directory because download_and_extract_dataset expects the parent.
                if os.path.basename(root) == 'training2017':
                    resolved_path = os.path.dirname(root)
                else:
                    resolved_path = root
                print(f"Found dataset at: {resolved_path}")
                return resolved_path
        print(f"REFERENCE.csv not found in {KAGGLE_INPUT_DIR}. Using default Kaggle path.")
        return KAGGLE_INPUT_DIR

    # 2. Check dataset_path.txt
    if os.path.exists(DATASET_PATH_FILE):
        try:
            with open(DATASET_PATH_FILE, 'r') as f:
                path = f.read().strip()
                if path:
                    print(f"Loaded dataset path from '{DATASET_PATH_FILE}': {path}")
                    return path
        except Exception as e:
            print(f"Error reading {DATASET_PATH_FILE}: {e}")
    else:
        # Create dataset_path.txt with DEFAULT_DATA_ROOT as default value
        try:
            with open(DATASET_PATH_FILE, 'w') as f:
                f.write(DEFAULT_DATA_ROOT)
            print(f"Created '{DATASET_PATH_FILE}' with default path: {DEFAULT_DATA_ROOT}")
        except Exception as e:
            print(f"Could not create {DATASET_PATH_FILE}: {e}")

    # 3. Fallback to DEFAULT_DATA_ROOT
    print(f"Using default local data root: {DEFAULT_DATA_ROOT}")
    return DEFAULT_DATA_ROOT
