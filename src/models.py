import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import inv, eigh
import config as cfg

class SqueezeExcitation1D(nn.Module):
    """
    1D Squeeze-and-Excitation (SE) block for channel attention.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Ensure reduction does not reduce channels to 0
        reduced_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, C, L]
        b, c, _ = x.size()
        y = x.mean(dim=-1) # Global Average Pooling -> [B, C]
        y = self.fc(y).view(b, c, 1) # [B, C, 1]
        return x * y

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, stride=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout1d(dropout_rate)
        
    def forward(self, x):
        return self.dropout(self.activation(self.bn(self.conv(x))))

class MultiScaleConvBlock1D(nn.Module):
    """
    Multi-Scale Convolutional Block.
    Extracts features using multiple kernel sizes (3, 7, 15) in parallel
    to capture both narrow QRS complexes and wide P/T-waves.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        # Parallel convolutions with kernel sizes 3, 7, 15
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.conv15 = nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7)
        
        # Merge outputs
        self.proj = nn.Conv1d(out_channels * 3, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout1d(dropout_rate)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        c3 = self.conv3(x)
        c7 = self.conv7(x)
        c15 = self.conv15(x)
        merged = torch.cat([c3, c7, c15], dim=1)
        return self.pool(self.dropout(self.activation(self.bn(self.proj(merged)))))

class ResidualConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, use_se=False):
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
            
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation1D(out_channels)
            
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.conv_blocks(x)
        shortcut = self.shortcut(x)
        out = out + shortcut
        if self.use_se:
            out = self.se(out)
        out = self.pool(out)
        return out

class AttentionPooling(nn.Module):
    """
    Attention pooling layer to focus on irregular rhythm regions (like AF) across time.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, input_dim]
        # mask: [batch_size, seq_len]
        attn_scores = self.attn(x).squeeze(-1) # [batch_size, seq_len]
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1) # [batch_size, seq_len, 1]
        context = torch.sum(x * attn_weights, dim=1) # [batch_size, input_dim]
        return context

class ResNetGRUFeatureExtractor(nn.Module):
    """
    1D-ResNet followed by BiGRU/BiLSTM to extract deep features from ECG raw signals.
    """
    def __init__(self, in_channels=1, gru_hidden=128, dropout_rate=0.1):
        super().__init__()
        # Gradual filter count scaling: 16 -> 32 -> 64 -> 128
        self.blocks = nn.Sequential(
            # Block 1: Multi-scale features (16 filters)
            MultiScaleConvBlock1D(in_channels, 16, dropout_rate),
            ResidualConvBlock1D(16, 16, dropout_rate, use_se=cfg.USE_SE_BLOCK),
            # Blocks 3 and 4: 32 filters
            ResidualConvBlock1D(16, 32, dropout_rate, use_se=cfg.USE_SE_BLOCK),
            ResidualConvBlock1D(32, 32, dropout_rate, use_se=cfg.USE_SE_BLOCK),
            # Blocks 5 and 6: 64 filters
            ResidualConvBlock1D(32, 64, dropout_rate, use_se=cfg.USE_SE_BLOCK),
            ResidualConvBlock1D(64, 64, dropout_rate, use_se=cfg.USE_SE_BLOCK),
            # Blocks 7 and 8: 128 filters (dynamic expansion)
            ResidualConvBlock1D(64, 128, dropout_rate, use_se=cfg.USE_SE_BLOCK),
            ResidualConvBlock1D(128, 128, dropout_rate, use_se=cfg.USE_SE_BLOCK)
        )
        
        # Configure Recurrent layer based on configs
        if cfg.RNN_TYPE == 'BiLSTM':
            self.rnn = nn.LSTM(
                input_size=128,
                hidden_size=gru_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
        else:
            self.rnn = nn.GRU(
                input_size=128,
                hidden_size=gru_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            
        self.use_attention_pooling = cfg.USE_ATTENTION_POOLING
        if self.use_attention_pooling:
            self.attn_pooling = AttentionPooling(gru_hidden * 2)
            
        self.out_dim = gru_hidden * 2
        
    def forward(self, x, mask=None):
        # x: [batch_size, 1, seq_len]
        features = self.blocks(x) # [batch_size, 128, downsampled_seq_len]
        features = features.permute(0, 2, 1) # [batch_size, downsampled_seq_len, 128]
        
        # Pack padded sequence if mask is provided
        if mask is not None:
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
            rnn_out, _ = self.rnn(packed_features)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=features.shape[1])
            
            if self.use_attention_pooling:
                last_hidden = self.attn_pooling(rnn_out, downsampled_mask)
            else:
                # Extract last non-padded hidden state
                batch_size = rnn_out.size(0)
                idx = (lengths - 1).view(-1, 1, 1).expand(batch_size, 1, rnn_out.size(2)).to(rnn_out.device)
                last_hidden = rnn_out.gather(1, idx).squeeze(1)
        else:
            rnn_out, _ = self.rnn(features)
            if self.use_attention_pooling:
                last_hidden = self.attn_pooling(rnn_out, None)
            else:
                last_hidden = rnn_out[:, -1, :] # Last step
            
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
            Sw += (Xc_class.T @ np.ones((Xc_class.shape[0], Yc_class.shape[0])) @ Yc_class) / (np.sum(idx) * N)
            
        # Sb (Inter-class correlation matrix) = St - Sw
        Sb = St - Sw
        
        # S_tilde_xy = Sw - eta * Sb
        S_tilde_xy = Sw - self.eta * Sb
        
        # Solve generalized eigenvalue problem
        eigvals_x, eigvecs_x = eigh(Sxx)
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
    Classifier that takes fused features and performs classification.
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

class AttentionFusionClassifier(nn.Module):
    """
    Fuses Traditional (X) and Deep (Y) features using a Gated Cross-Attention mechanism.
    """
    def __init__(self, trad_dim, deep_dim, num_classes=4):
        super().__init__()
        # Project both to same dimension for attention
        self.proj_trad = nn.Linear(trad_dim, 64)
        self.proj_deep = nn.Linear(deep_dim, 64)
        
        # Cross-attention weights
        self.query = nn.Linear(64, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        
        # Gated fusion gate
        self.gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x_trad, x_deep):
        # x_trad: [B, trad_dim], x_deep: [B, deep_dim]
        t = self.proj_trad(x_trad)  # [B, 64]
        d = self.proj_deep(x_deep)  # [B, 64]
        
        # We treat t and d as sequence of length 2
        features = torch.stack([t, d], dim=1) # [B, 2, 64]
        
        q = self.query(features) # [B, 2, 64]
        k = self.key(features)   # [B, 2, 64]
        v = self.value(features) # [B, 2, 64]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(64.0) # [B, 2, 2]
        attn_weights = torch.softmax(scores, dim=-1) # [B, 2, 2]
        fused_features = torch.matmul(attn_weights, v) # [B, 2, 64]
        
        # Flatten and gate
        fused_flat = fused_features.reshape(-1, 128)
        g = self.gate(fused_flat) # [B, 64]
        
        # Gated sum of projections
        output = g * t + (1 - g) * d
        
        return self.classifier(output)
