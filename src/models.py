import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import inv, eigh

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, stride=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout1d(dropout_rate)
        
    def forward(self, x):
        return self.dropout(self.activation(self.bn(self.conv(x))))

class ResidualConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock1D(in_channels, out_channels, kernel_size=3, dropout_rate=dropout_rate),
            ConvBlock1D(out_channels, out_channels, kernel_size=3, dropout_rate=dropout_rate),
            ConvBlock1D(out_channels, out_channels, kernel_size=3, dropout_rate=dropout_rate),
            ConvBlock1D(out_channels, out_channels, kernel_size=3, dropout_rate=dropout_rate)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.conv_blocks(x)
        shortcut = self.shortcut(x)
        out = out + shortcut
        out = self.pool(out)
        return out

class ResNetGRUFeatureExtractor(nn.Module):
    """
    1D-ResNet followed by GRU to extract deep features from ECG raw signals.
    """
    def __init__(self, in_channels=1, gru_hidden=64, dropout_rate=0.1):
        super().__init__()
        self.blocks = nn.Sequential(
            # Blocks 1 and 2: 16 filters
            ResidualConvBlock1D(in_channels, 16, dropout_rate),
            ResidualConvBlock1D(16, 16, dropout_rate),
            # Blocks 3 and 4: 32 filters
            ResidualConvBlock1D(16, 32, dropout_rate),
            ResidualConvBlock1D(32, 32, dropout_rate),
            # Blocks 5 and 6: 64 filters
            ResidualConvBlock1D(32, 64, dropout_rate),
            ResidualConvBlock1D(64, 64, dropout_rate)
        )
        
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True # Bidirectional to capture both directions, 2 * gru_hidden output
        )
        
        self.out_dim = gru_hidden * 2
        
    def forward(self, x, mask=None):
        # x: [batch_size, 1, seq_len]
        features = self.blocks(x) # [batch_size, 64, downsampled_seq_len]
        features = features.permute(0, 2, 1) # [batch_size, downsampled_seq_len, 64]
        
        # Pack padded sequence if mask is provided
        if mask is not None:
            # Downsample the mask to match features seq_len
            # Average pooling of the mask over same stride (2^6 = 64)
            # Or simpler: downsample mask by slicing/resizing
            downsample_factor = x.shape[-1] // features.shape[1]
            downsampled_mask = mask[:, ::downsample_factor]
            # Match lengths exactly
            if downsampled_mask.shape[1] > features.shape[1]:
                downsampled_mask = downsampled_mask[:, :features.shape[1]]
            elif downsampled_mask.shape[1] < features.shape[1]:
                # Pad mask with zeros
                pad = torch.zeros(downsampled_mask.shape[0], features.shape[1] - downsampled_mask.shape[1], device=mask.device)
                downsampled_mask = torch.cat([downsampled_mask, pad], dim=1)
                
            lengths = downsampled_mask.sum(dim=1).clamp(min=1).cpu().long()
            
            # Pack sequence
            packed_features = nn.utils.rnn.pack_padded_sequence(
                features, lengths, batch_first=True, enforce_sorted=False
            )
            gru_out, _ = self.gru(packed_features)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, total_length=features.shape[1])
            
            # Extract last non-padded hidden state
            batch_size = gru_out.size(0)
            idx = (lengths - 1).view(-1, 1, 1).expand(batch_size, 1, gru_out.size(2)).to(gru_out.device)
            last_hidden = gru_out.gather(1, idx).squeeze(1)
        else:
            gru_out, _ = self.gru(features)
            last_hidden = gru_out[:, -1, :] # Last step
            
        return last_hidden

class DCCAProjection:
    """
    Discriminant Canonical Correlation Analysis (DCCA) for feature fusion.
    Fuses two feature spaces X (Traditional) and Y (Deep Learning) using class labels.
    """
    def __init__(self, d_components=16, eta=0.1, reg=1e-4):
        self.d_components = d_components
        self.eta = eta
        self.reg = reg
        self.Wx = None
        self.Wy = None
        self.mean_x = None
        self.mean_y = None

    def fit(self, X, Y, labels):
        """
        Fits DCCA projection matrices.
        X: numpy array [N, p] (traditional features)
        Y: numpy array [N, q] (deep features)
        labels: numpy array [N] (class labels 0..C-1)
        """
        N, p = X.shape
        _, q = Y.shape
        
        # Center features
        self.mean_x = np.mean(X, axis=0)
        self.mean_y = np.mean(Y, axis=0)
        Xc = X - self.mean_x
        Yc = Y - self.mean_y
        
        # Total covariance / correlation matrices with regularization
        Sxx = (Xc.T @ Xc) / (N - 1) + self.reg * np.eye(p)
        Syy = (Yc.T @ Yc) / (N - 1) + self.reg * np.eye(q)
        St = (Xc.T @ Yc) / (N - 1) # Total cross-correlation
        
        # Calculate Sw (Intra-class correlation matrix)
        classes = np.unique(labels)
        Sw = np.zeros((p, q))
        
        for c in classes:
            idx = (labels == c)
            if np.sum(idx) <= 1:
                continue
            Xc_class = Xc[idx]
            Yc_class = Yc[idx]
            # Sw_c = X_c * 1 * Y_c^T
            # Normalizing by number of class samples to prevent bias towards large classes
            Sw += (Xc_class.T @ np.ones((Xc_class.shape[0], Yc_class.shape[0])) @ Yc_class) / (np.sum(idx) * N)
            
        # Sb (Inter-class correlation matrix) = St - Sw
        Sb = St - Sw
        
        # S_tilde_xy = Sw - eta * Sb
        S_tilde_xy = Sw - self.eta * Sb
        
        # Solve generalized eigenvalue problem:
        # We want to find projection directions Wx and Wy using SVD of:
        # T = Sxx^-1/2 * S_tilde_xy * Syy^-1/2
        # Let's perform symmetric square root of Sxx and Syy
        eigvals_x, eigvecs_x = eigh(Sxx)
        # Avoid negative/zero eigenvalues due to precision
        eigvals_x = np.clip(eigvals_x, a_min=1e-12, a_max=None)
        Sxx_m12 = eigvecs_x @ np.diag(1.0 / np.sqrt(eigvals_x)) @ eigvecs_x.T
        
        eigvals_y, eigvecs_y = eigh(Syy)
        eigvals_y = np.clip(eigvals_y, a_min=1e-12, a_max=None)
        Syy_m12 = eigvecs_y @ np.diag(1.0 / np.sqrt(eigvals_y)) @ eigvecs_y.T
        
        T = Sxx_m12 @ S_tilde_xy @ Syy_m12
        
        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(T)
        V = Vt.T
        
        # Keep first d components
        d = min(self.d_components, p, q)
        self.Wx = Sxx_m12 @ U[:, :d]
        self.Wy = Syy_m12 @ V[:, :d]
        
        return self

    def transform(self, X, Y):
        """
        Projects X and Y onto the DCCA space.
        Returns:
            Z_f: Concatenated features [N, 2 * d_components]
        """
        Xc = X - self.mean_x
        Yc = Y - self.mean_y
        
        Zx = Xc @ self.Wx
        Zy = Yc @ self.Wy
        
        # Concatenate projected features in series (Tandem method)
        Z_f = np.hstack((Zx, Zy))
        return Z_f

class FusionClassifier(nn.Module):
    """
    Classifier that takes fused DCCA features and performs classification.
    """
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)
