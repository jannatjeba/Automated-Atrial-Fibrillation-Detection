import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset

class PhysioNet2017Dataset(Dataset):
    """
    Dataset class to load, preprocess, and serve ECG recordings from the
    PhysioNet/CinC Challenge 2017 dataset.
    """
    LABEL_MAP = {'N': 0, 'A': 1, 'O': 2, '~': 3}
    REV_LABEL_MAP = {0: 'Normal', 1: 'AF', 2: 'Other', 3: 'Noise'}

    def __init__(self, data_dir, csv_path, target_fs=300, max_len_sec=30, 
                 preprocess=True, mode='pad', segment_len_sec=10, overlap_sec=5):
        """
        Args:
            data_dir (str): Directory containing the extracted .mat files.
            csv_path (str): Path to REFERENCE.csv containing labels.
            target_fs (int): Target sampling rate (default 300Hz).
            max_len_sec (int): Maximum signal length in seconds for padding.
            preprocess (bool): Whether to apply Butterworth bandpass filter.
            mode (str): 'pad' (pad/truncate to max_len_sec) or 'segment' (split into overlapping windows).
            segment_len_sec (int): Window length for segment mode.
            overlap_sec (int): Overlap length for segment mode.
        """
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.target_fs = target_fs
        self.max_len = target_fs * max_len_sec
        self.preprocess = preprocess
        self.mode = mode
        self.segment_len = target_fs * segment_len_sec
        self.overlap = target_fs * overlap_sec
        
        # Load reference labels
        self.df = pd.read_csv(csv_path, header=None, names=['record', 'label'])
        self.df['label_idx'] = self.df['label'].map(self.LABEL_MAP)
        
        # In segment mode, we pre-generate pointers to individual segments
        self.records = []
        if self.mode == 'segment':
            self._generate_segments()
        else:
            self.records = self.df.to_dict('records')

    def _generate_segments(self):
        """Generates segment mappings for sliding window mode."""
        for _, row in self.df.iterrows():
            record_name = row['record']
            label_idx = row['label_idx']
            mat_path = os.path.join(self.data_dir, f"{record_name}.mat")
            
            if not os.path.exists(mat_path):
                continue
                
            try:
                # Get signal length quickly without loading full mat file if possible, 
                # or just load it since 2017 dataset is small
                mat_data = sio.loadmat(mat_path)
                signal = mat_data['valval' if 'valval' in mat_data else 'val'][0]
                sig_len = len(signal)
                
                if sig_len < self.segment_len:
                    # Pad short signals
                    self.records.append({
                        'record': record_name,
                        'label_idx': label_idx,
                        'start': 0,
                        'end': sig_len,
                        'need_padding': True
                    })
                else:
                    step = self.segment_len - self.overlap
                    for start in range(0, sig_len - self.segment_len + 1, step):
                        self.records.append({
                            'record': record_name,
                            'label_idx': label_idx,
                            'start': start,
                            'end': start + self.segment_len,
                            'need_padding': False
                        })
            except Exception as e:
                print(f"Error reading {record_name}: {e}")

    @staticmethod
    def butter_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=300.0, order=3):
        """Applies a Butterworth bandpass filter to denoise the ECG signal."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    def _load_signal(self, record_name):
        mat_path = os.path.join(self.data_dir, f"{record_name}.mat")
        mat_data = sio.loadmat(mat_path)
        signal = mat_data['valval' if 'valval' in mat_data else 'val'][0]
        # PhysioNet signals are stored as integers; convert to float mV
        signal = signal.astype(np.float32)
        
        # Denoising
        if self.preprocess:
            signal = self.butter_bandpass_filter(signal, lowcut=0.5, highcut=45.0, fs=self.target_fs)
            
        # Standardize signal
        mean_val = np.mean(signal)
        std_val = np.std(signal) + 1e-8
        signal = (signal - mean_val) / std_val
        
        return signal

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_info = self.records[idx]
        record_name = record_info['record']
        label = record_info['label_idx']
        
        signal = self._load_signal(record_name)
        
        if self.mode == 'segment':
            start = record_info['start']
            end = record_info['end']
            sig_segment = signal[start:end]
            
            if record_info['need_padding']:
                # Pad to segment length
                padded = np.zeros(self.segment_len, dtype=np.float32)
                padded[:len(sig_segment)] = sig_segment
                attention_mask = np.zeros(self.segment_len, dtype=np.float32)
                attention_mask[:len(sig_segment)] = 1.0
                sig_segment = padded
            else:
                attention_mask = np.ones(self.segment_len, dtype=np.float32)
                
            return (
                torch.tensor(sig_segment, dtype=torch.float32).unsqueeze(0),
                torch.tensor(attention_mask, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long),
                record_name
            )
            
        else: # 'pad' mode
            sig_len = len(signal)
            attention_mask = np.zeros(self.max_len, dtype=np.float32)
            
            if sig_len < self.max_len:
                padded = np.zeros(self.max_len, dtype=np.float32)
                padded[:sig_len] = signal
                attention_mask[:sig_len] = 1.0
                signal = padded
            else:
                signal = signal[:self.max_len]
                attention_mask[:] = 1.0
                
            return (
                torch.tensor(signal, dtype=torch.float32).unsqueeze(0),
                torch.tensor(attention_mask, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long),
                record_name
            )

def download_and_extract_dataset(dest_dir):
    """Downloads and extracts the PhysioNet CinC 2017 dataset if not already present."""
    # Check if REFERENCE.csv is directly in dest_dir (e.g. Kaggle datasets flat structure)
    direct_csv_path = os.path.join(dest_dir, "REFERENCE.csv")
    if os.path.exists(direct_csv_path):
        print(f"REFERENCE.csv found directly in {dest_dir}. Bypassing download/extraction.")
        return dest_dir, direct_csv_path

    # Check if REFERENCE.csv is in training2017 subdirectory
    csv_path = os.path.join(dest_dir, "training2017", "REFERENCE.csv")
    data_dir = os.path.join(dest_dir, "training2017")
    if os.path.exists(csv_path):
        print(f"REFERENCE.csv found in {data_dir}. Bypassing download/extraction.")
        return data_dir, csv_path

    # Otherwise, download and extract
    url = "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip"
    zip_path = os.path.join(dest_dir, "training2017.zip")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    if not os.path.exists(zip_path):
        print(f"Downloading PhysioNet CinC 2017 dataset from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
        
    extract_path = os.path.join(dest_dir, "training2017")
    if not os.path.exists(extract_path) or len(os.listdir(extract_path)) < 100:
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print("Extraction complete.")
        
    return data_dir, csv_path
