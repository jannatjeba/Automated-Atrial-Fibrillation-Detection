import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis

def detect_r_peaks(signal, fs=300):
    """
    Detects R-peaks in ECG signal using a simplified Pan-Tompkins algorithm.
    """
    if len(signal) < 2 * fs:
        return np.array([])
        
    # Step 1: Bandpass filter is already applied during preprocessing
    # Step 2: Differentiate the signal
    diff_sig = np.diff(signal)
    
    # Step 3: Square the signal
    squared_sig = diff_sig ** 2
    
    # Step 4: Moving window integration
    window_len = int(0.12 * fs)
    integrated_sig = np.convolve(squared_sig, np.ones(window_len)/window_len, mode='same')
    
    # Step 5: Peak detection with thresholding
    min_dist = int(0.3 * fs)  # Min refractory period (300ms)
    
    # Adaptive threshold based on the signal amplitude distribution
    thresh = np.percentile(integrated_sig, 90) * 0.35
    if thresh <= 0:
        thresh = np.mean(integrated_sig)
        
    peaks, _ = find_peaks(integrated_sig, distance=min_dist, height=thresh)
    
    # Map peaks from integrated signal back to local maxima in original signal
    r_peaks = []
    search_win = int(0.1 * fs)
    for p in peaks:
        left = max(0, p - search_win)
        right = min(len(signal), p + search_win)
        if left >= right:
            continue
        r_peak = left + np.argmax(signal[left:right])
        r_peaks.append(r_peak)
        
    return np.unique(r_peaks)

def calculate_sampen(x, m=2, r=0.2):
    """
    Fast Sample Entropy (SampEn) implementation using numpy.
    """
    N = len(x)
    if N < m + 1:
        return 0.0
    
    r_threshold = r * np.std(x)
    if r_threshold == 0:
        r_threshold = 1e-6
        
    def _phi(m_val):
        # Create overlapping sub-sequences of length m_val
        x_mat = np.array([x[i:i + m_val] for i in range(N - m_val + 1)])
        # Compute pairwise distances using Chebyshev distance
        # For large N, we do it in batches to avoid O(N^2) memory footprint
        num_sub = len(x_mat)
        count = 0
        for i in range(num_sub):
            diffs = np.max(np.abs(x_mat - x_mat[i]), axis=1)
            count += np.sum(diffs < r_threshold) - 1 # Exclude self-comparison
        return count / (num_sub * (num_sub - 1) + 1e-8)
        
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    
    if phi_m == 0 or phi_m1 == 0:
        return 0.0
        
    return -np.log(phi_m1 / phi_m)

def get_segment_stats(arr, num_segments=6):
    """
    Divides an array into equal segments and calculates mean, var, skew, and kurtosis.
    """
    stats = []
    if len(arr) < num_segments:
        # Fallback if array is too small: pad with mean or zeros
        val = np.mean(arr) if len(arr) > 0 else 0.0
        for _ in range(num_segments):
            stats.extend([val, 0.0, 0.0, 0.0])
        return stats
        
    seg_size = len(arr) // num_segments
    for i in range(num_segments):
        start = i * seg_size
        end = (i + 1) * seg_size if i < num_segments - 1 else len(arr)
        seg = arr[start:end]
        
        m = np.mean(seg)
        v = np.var(seg)
        s = skew(seg) if len(seg) > 2 else 0.0
        k = kurtosis(seg) if len(seg) > 2 else 0.0
        
        # Replace NaNs/Infs
        s = 0.0 if np.isnan(s) or np.isinf(s) else s
        k = 0.0 if np.isnan(k) or np.isinf(k) else k
        
        stats.extend([m, v, s, k])
    return stats

def extract_features(signal, fs=300):
    """
    Extracts all traditional handcrafted features from ECG signal.
    """
    features = []
    
    # 1. R-peaks detection
    r_peaks = detect_r_peaks(signal, fs)
    
    # --- RR-Interval (RRI) Features ---
    if len(r_peaks) >= 2:
        rris = np.diff(r_peaks) / fs
        mean_rri = np.mean(rris)
        std_rri = np.std(rris)
        var_rri = np.var(rris)
        max_rri = np.max(rris)
        min_rri = np.min(rris)
        
        # RMSSD and SDSD
        diff_rris = np.diff(rris)
        rmssd = np.sqrt(np.mean(diff_rris ** 2)) if len(diff_rris) > 0 else 0.0
        sdsd = np.std(diff_rris) if len(diff_rris) > 0 else 0.0
        
        # pNN50
        nn50 = np.sum(np.abs(diff_rris) > 0.05) if len(diff_rris) > 0 else 0.0
        pnn50 = nn50 / len(rris) if len(rris) > 0 else 0.0
        
        # Segment stats of RRIs
        rri_seg_stats = get_segment_stats(rris, num_segments=6)
    else:
        # Fallback values
        mean_rri, std_rri, var_rri, max_rri, min_rri = 0.8, 0.0, 0.0, 0.8, 0.8
        rmssd, sdsd, pnn50 = 0.0, 0.0, 0.0
        rri_seg_stats = [0.8, 0.0, 0.0, 0.0] * 6
        rris = np.array([0.8])
        
    features.extend([mean_rri, std_rri, var_rri, max_rri, min_rri, rmssd, sdsd, pnn50])
    features.extend(rri_seg_stats)
    
    # --- P-wave Features ---
    # Paper extracting P-wave stats. P-wave is typically 200ms to 50ms before R-peak.
    # In AF, P-wave is absent, representing f-wave noise.
    p_waves = []
    p_wave_duration = int(0.15 * fs) # 150ms window
    p_wave_offset = int(0.05 * fs)   # 50ms before R-peak
    
    for r in r_peaks:
        start = r - p_wave_duration - p_wave_offset
        end = r - p_wave_offset
        if start >= 0 and end < len(signal):
            p_waves.append(signal[start:end])
            
    if len(p_waves) > 0:
        p_waves_flat = np.concatenate(p_waves)
        mean_p = np.mean(p_waves_flat)
        var_p = np.var(p_waves_flat)
        skew_p = skew(p_waves_flat)
        kurt_p = kurtosis(p_waves_flat)
        
        # Clean NaNs
        skew_p = 0.0 if np.isnan(skew_p) or np.isinf(skew_p) else skew_p
        kurt_p = 0.0 if np.isnan(kurt_p) or np.isinf(kurt_p) else kurt_p
        
        # Sample Entropy of P-waves (calculated on concatenated p-waves, limited to 1000 samples for speed)
        sampen_p = calculate_sampen(p_waves_flat[:1000], m=2, r=0.2)
        sampen_coeff_p = sampen_p / (np.std(p_waves_flat) + 1e-8)
        
        # 6 segments of P-wave stats (Mean, Var, Skew)
        p_seg_stats = []
        seg_size = len(p_waves_flat) // 6
        if seg_size > 2:
            for i in range(6):
                start = i * seg_size
                end = (i + 1) * seg_size if i < 5 else len(p_waves_flat)
                seg = p_waves_flat[start:end]
                p_seg_stats.extend([np.mean(seg), np.var(seg), skew(seg)])
        else:
            p_seg_stats = [mean_p, var_p, skew_p] * 6
            
    else:
        mean_p, var_p, skew_p, kurt_p, sampen_p, sampen_coeff_p = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        p_seg_stats = [0.0, 0.0, 0.0] * 6
        
    features.extend([mean_p, var_p, skew_p, kurt_p, sampen_p, sampen_coeff_p])
    # Ensure p_seg_stats contains exactly 18 elements
    p_seg_stats = p_seg_stats[:18]
    if len(p_seg_stats) < 18:
        p_seg_stats.extend([0.0] * (18 - len(p_seg_stats)))
    features.extend(p_seg_stats)
    
    # --- Frequency Domain and Signal Processing Features ---
    # Power Spectral Density (PSD) via Welch
    nperseg = min(256, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    
    # Calculate energy in specific bands: 0.1-6Hz, 6-12Hz, 12-20Hz, 20-30Hz, 30-45Hz
    def get_band_power(f_min, f_max):
        idx = np.where((freqs >= f_min) & (freqs <= f_max))[0]
        if len(idx) == 0:
            return 0.0
        return np.trapz(psd[idx], freqs[idx])
        
    bp_1 = get_band_power(0.1, 6.0)
    bp_2 = get_band_power(6.0, 12.0)
    bp_3 = get_band_power(12.0, 20.0)
    bp_4 = get_band_power(20.0, 30.0)
    bp_5 = get_band_power(30.0, 45.0)
    
    features.extend([bp_1, bp_2, bp_3, bp_4, bp_5])
    
    # Median Absolute Deviation (MAD) of the signal
    mad = np.median(np.abs(signal - np.median(signal)))
    
    # Coefficient of Variation of RRI
    cv_rri = std_rri / (mean_rri + 1e-8)
    
    # Heart Rate Variability (HRV) - approximate representation using SDNN (std_rri)
    hrv = std_rri
    
    # Sample Entropy of the entire signal (sub-sampled to 1000 pts for speed)
    sampen_signal = calculate_sampen(signal[::max(1, len(signal)//1000)], m=2, r=0.2)
    
    features.extend([mad, cv_rri, hrv, sampen_signal])
    
    # Clean all features for NaNs and Infs
    features = [0.0 if np.isnan(f) or np.isinf(f) else f for f in features]
    
    return np.array(features, dtype=np.float32)
