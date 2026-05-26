import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, detrend
import torch
from torch.utils.data import Dataset
import pywt
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg

class PhysioNet2017Dataset(Dataset):
    """
    Dataset class to load, preprocess, and serve ECG recordings from the
    PhysioNet/CinC Challenge 2017 dataset.
    """
    LABEL_MAP = {'N': 0, 'A': 1, 'O': 2, '~': 3}
    REV_LABEL_MAP = {0: 'Normal', 1: 'AF', 2: 'Other', 3: 'Noise'}

    def __init__(self, data_dir, csv_path, target_fs=300, max_len_sec=30, 
                 preprocess=True, mode='pad', segment_len_sec=10, overlap_sec=5, is_train=False):
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
            is_train (bool): True if training (enables data augmentations).
        """
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.target_fs = target_fs
        self.max_len = target_fs * max_len_sec
        self.preprocess = preprocess
        self.mode = mode
        self.segment_len = target_fs * segment_len_sec
        self.overlap = target_fs * overlap_sec
        self.is_train = is_train
        
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

    @staticmethod
    def wavelet_denoise(signal, wavelet='db4', level=9):
        """Applies wavelet denoising (soft thresholding) on ECG signals."""
        coeff = pywt.wavedec(signal, wavelet, mode="per")
        # Median Absolute Deviation (MAD) of detail coefficients for noise estimation
        mad = np.median(np.abs(coeff[-1] - np.median(coeff[-1])))
        if mad == 0 or np.isnan(mad):
            return signal
        sigma = (1/0.6745) * mad
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
        coeff[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:]]
        return pywt.waverec(coeff, wavelet, mode="per")[:len(signal)]

    @staticmethod
    def frequency_mask(signal, max_mask_pct=0.05):
        """Randomly masks a band of frequencies in the frequency domain (SpecAugment for 1D)."""
        n = len(signal)
        fft_vals = np.fft.rfft(signal)
        n_freqs = len(fft_vals)
        
        mask_width = int(np.random.uniform(0.01, max_mask_pct) * n_freqs)
        if mask_width > 0:
            start_idx = np.random.randint(0, n_freqs - mask_width)
            fft_vals[start_idx:start_idx + mask_width] = 0.0
            
        return np.fft.irfft(fft_vals, n)

    @staticmethod
    def safe_reflect_pad(signal, target_len):
        """Safely pads a signal using reflection padding, tiling first if the signal is too short."""
        sig_len = len(signal)
        if sig_len >= target_len:
            return signal[:target_len]
        pad_len = target_len - sig_len
        if pad_len > sig_len:
            repeats = (target_len // sig_len) + 1
            signal = np.tile(signal, repeats)
            sig_len = len(signal)
            pad_len = target_len - sig_len
            if pad_len <= 0:
                return signal[:target_len]
        return np.pad(signal, (0, pad_len), mode='reflect')

    def _load_signal(self, record_name):
        mat_path = os.path.join(self.data_dir, f"{record_name}.mat")
        mat_data = sio.loadmat(mat_path)
        signal = mat_data['valval' if 'valval' in mat_data else 'val'][0]
        # PhysioNet signals are stored as integers; convert to float mV
        signal = signal.astype(np.float32)
        
        # Denoising
        if self.preprocess:
            # 1. Baseline wander removal (Scipy Detrend)
            if cfg.DETREND_SIGNAL:
                signal = detrend(signal)
                
            # 2. Bandpass filtering
            signal = self.butter_bandpass_filter(signal, lowcut=0.5, highcut=45.0, fs=self.target_fs)
            
            # 3. Wavelet denoising
            if cfg.WAVELET_DENOISING:
                try:
                    signal = self.wavelet_denoise(signal)
                except Exception:
                    pass
            
        # Standardize signal
        mean_val = np.mean(signal)
        std_val = np.std(signal) + 1e-8
        signal = (signal - mean_val) / std_val
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        return signal

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_info = self.records[idx]
        record_name = record_info['record']
        label = record_info['label_idx']
        
        signal = self._load_signal(record_name)
        
        # Apply training-specific data augmentations on the raw signal
        if self.is_train and cfg.USE_DATA_AUGMENTATION:
            # Random Crop (before padding)
            if len(signal) > self.max_len:
                start = np.random.randint(0, len(signal) - self.max_len)
                signal = signal[start:start + self.max_len]
            
            # Add Gaussian noise (50% probability)
            if np.random.rand() < 0.5:
                noise = np.random.normal(0, 0.05, len(signal))
                signal = signal + noise
                
            # Scaling (50% probability)
            if np.random.rand() < 0.5:
                scaling_factor = np.random.uniform(0.8, 1.2)
                signal = signal * scaling_factor
                
            # Time Shift / Rolling (50% probability)
            if np.random.rand() < 0.5:
                shift = np.random.randint(-int(0.2 * self.target_fs), int(0.2 * self.target_fs))
                signal = np.roll(signal, shift)
                
            # Frequency Domain Spectral Masking (50% probability)
            if np.random.rand() < 0.5:
                signal = self.frequency_mask(signal)
        
        if self.mode == 'segment':
            start = record_info['start']
            end = record_info['end']
            sig_segment = signal[start:end]
            
            if record_info['need_padding']:
                # Pad to segment length
                if cfg.PADDING_MODE == 'reflect':
                    padded = self.safe_reflect_pad(sig_segment, self.segment_len)
                else:
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
                if cfg.PADDING_MODE == 'reflect':
                    padded = self.safe_reflect_pad(signal, self.max_len)
                else:
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
