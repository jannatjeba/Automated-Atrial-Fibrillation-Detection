import os
import shutil
import numpy as np
import pandas as pd
import scipy.io as sio
import torch

from src.dataset import PhysioNet2017Dataset
from src.train import main

class Args:
    def __init__(self):
        self.data_root = "./mock_data"
        self.folds = 2
        self.max_len_sec = 10  # 10s for fast test
        self.gru_hidden = 16   # small size
        self.dcca_dim = 4      # small DCCA dimension
        self.dcca_eta = 0.1
        self.batch_size = 4
        self.deep_epochs = 1
        self.classifier_epochs = 1
        self.seed = 42

def create_mock_dataset(data_root):
    """
    Creates a small mock dataset of 8 files representing ECG recordings.
    """
    train_dir = os.path.join(data_root, "training2017")
    os.makedirs(train_dir, exist_ok=True)
    
    # 8 records, 2 for each of the 4 classes
    records = [
        ('A0001', 'N'), ('A0002', 'N'),
        ('A0003', 'A'), ('A0004', 'A'),
        ('A0005', 'O'), ('A0006', 'O'),
        ('A0007', '~'), ('A0008', '~')
    ]
    
    fs = 300
    duration = 10 # 10 seconds
    t = np.linspace(0, duration, duration * fs, endpoint=False)
    
    # Create mock signal and save as .mat
    for name, label in records:
        # Mix some sine waves as mock ECG
        signal = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.cos(2 * np.pi * 10.0 * t)
        if label == 'A':
            # Add irregular waves for AF
            signal += 0.3 * np.sin(2 * np.pi * 7.5 * t)
        elif label == '~':
            # High noise
            signal += np.random.normal(0, 1.5, len(t))
            
        # Convert to 16-bit int representation
        val = (signal * 1000).astype(np.int16)
        
        # Save mat
        sio.savemat(os.path.join(train_dir, f"{name}.mat"), {'val': val.reshape(1, -1)})
        
    # Save REFERENCE.csv
    ref_df = pd.DataFrame(records)
    ref_df.to_csv(os.path.join(train_dir, "REFERENCE.csv"), header=False, index=False)
    print(f"Created mock dataset in {train_dir}")

def test_pipeline():
    data_root = "./mock_data"
    if os.path.exists(data_root):
        shutil.rmtree(data_root)
        
    create_mock_dataset(data_root)
    
    args = Args()
    print("Starting pipeline test...")
    try:
        main(args)
        print("\nPipeline test PASSED successfully!")
    except Exception as e:
        print(f"\nPipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists(data_root):
            shutil.rmtree(data_root)
            print("Cleaned up mock data.")

if __name__ == "__main__":
    test_pipeline()
