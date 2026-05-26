import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import pickle

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from dataset import download_and_extract_dataset, PhysioNet2017Dataset
from features import extract_features
from models import (
    ResNetGRUFeatureExtractor, 
    DCCAProjection, 
    FusionClassifier, 
    AttentionFusionClassifier
)
from utils import calculate_metrics

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FocalLoss(nn.Module):
    """
    Class-weighted Focal Loss to address extreme class imbalance.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights tensor
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def mixup_data(x, y, alpha=0.2, device='cpu'):
    """
    Returns mixed inputs, pairs of targets, and lambda for Mixup augmentation.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

def preextract_traditional_features(dataset, cache_path=None):
    """
    Pre-extracts traditional features for all records to speed up cross-validation.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached traditional features from {cache_path}...")
        data = np.load(cache_path)
        return data['features'], data['labels']
        
    print("Pre-extracting traditional handcrafted features (this may take a few minutes)...")
    all_features = []
    all_labels = []
    
    for idx in tqdm(range(len(dataset))):
        record_info = dataset.records[idx]
        record_name = record_info['record']
        label = record_info['label_idx']
        
        # Load and preprocess full signal (from validation dataset to keep it clean/non-augmented)
        signal = dataset._load_signal(record_name)
        
        # Extract features
        feats = extract_features(signal, fs=dataset.target_fs)
        all_features.append(feats)
        all_labels.append(label)
        
    all_features = np.array(all_features, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    
    if cache_path:
        np.savez(cache_path, features=all_features, labels=all_labels)
        print(f"Saved traditional features to {cache_path}")
        
    return all_features, all_labels

def train_deep_model(model, train_loader, val_loader, class_weights, epochs=20, device='cpu'):
    """
    Trains the ResNet + RNN model with a temporary classification head using advanced strategy.
    """
    classifier_head = nn.Linear(model.out_dim, 4).to(device)
    
    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier_head.parameters()), 
        lr=cfg.DEEP_LR, weight_decay=cfg.DEEP_WD
    )
    
    # Loss Function selection
    if cfg.USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=class_weights.to(device), gamma=cfg.FOCAL_GAMMA)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
    # Learning Rate Scheduler
    if cfg.USE_COSINE_SCHEDULER:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None
        
    # AMP Grad Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP)
    
    best_val_loss = float('inf')
    best_model_state = {
        'model': model.state_dict(),
        'classifier_head': classifier_head.state_dict()
    }
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        classifier_head.train()
        train_loss = 0.0
        
        for signals, masks, labels, _ in train_loader:
            signals, masks, labels = signals.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Apply Mixup (Data Augmentation)
            use_mixup = cfg.USE_DATA_AUGMENTATION and cfg.MIXUP_ALPHA > 0
            if use_mixup:
                mixed_signals, labels_a, labels_b, lam = mixup_data(signals, labels, cfg.MIXUP_ALPHA, device)
                perm = torch.randperm(signals.size(0)).to(device)
                mixed_masks = torch.max(masks, masks[perm])
                
                with torch.cuda.amp.autocast(enabled=cfg.USE_AMP):
                    deep_features = model(mixed_signals, mixed_masks)
                    outputs = classifier_head(deep_features)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                with torch.cuda.amp.autocast(enabled=cfg.USE_AMP):
                    deep_features = model(signals, masks)
                    outputs = classifier_head(deep_features)
                    loss = criterion(outputs, labels)
                    
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if cfg.USE_GRAD_CLIPPING:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(classifier_head.parameters()), 
                    max_norm=1.0
                )
                
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * signals.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Step Scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Validation
        model.eval()
        classifier_head.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for signals, masks, labels, _ in val_loader:
                signals, masks, labels = signals.to(device), masks.to(device), labels.to(device)
                
                with torch.cuda.amp.autocast(enabled=cfg.USE_AMP):
                    deep_features = model(signals, masks)
                    outputs = classifier_head(deep_features)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item() * signals.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        
        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model / Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {
                'model': model.state_dict(),
                'classifier_head': classifier_head.state_dict()
            }
        else:
            patience_counter += 1
            
        if cfg.USE_EARLY_STOPPING and patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break
            
    # Restore best weights
    model.load_state_dict(best_model_state['model'])
    return model

def extract_deep_features(model, dataloader, device):
    """
    Extracts deep learning features using the trained model.
    """
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for signals, masks, labels, _ in dataloader:
            signals, masks = signals.to(device), masks.to(device)
            with torch.cuda.amp.autocast(enabled=cfg.USE_AMP):
                deep_feat = model(signals, masks)
            features_list.append(deep_feat.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)

def train_fusion_classifier(classifier, Z_train, y_train, Z_val, y_val, class_weights, epochs=40, batch_size=64, device='cpu'):
    """
    Trains the Fusion Classifier on fused features.
    """
    optimizer = optim.Adam(classifier.parameters(), lr=cfg.CLASSIFIER_LR, weight_decay=cfg.CLASSIFIER_WD)
    
    # Loss Function selection
    if cfg.USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=class_weights.to(device), gamma=cfg.FOCAL_GAMMA)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
    # Set up DataLoader based on Fusion method
    if cfg.FUSION_METHOD == 'Attention':
        X_trad_train, Y_deep_train = Z_train
        X_trad_val, Y_deep_val = Z_val
        
        train_data = torch.utils.data.TensorDataset(
            torch.tensor(X_trad_train, dtype=torch.float32),
            torch.tensor(Y_deep_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
    else:
        train_data = torch.utils.data.TensorDataset(
            torch.tensor(Z_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    best_classifier_state = classifier.state_dict().copy()
    patience_counter = 0
    
    for epoch in range(epochs):
        classifier.train()
        if cfg.FUSION_METHOD == 'Attention':
            for batch_x, batch_y_deep, batch_labels in train_loader:
                batch_x, batch_y_deep, batch_labels = batch_x.to(device), batch_y_deep.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                outputs = classifier(batch_x, batch_y_deep)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
        else:
            for batch_z, batch_labels in train_loader:
                batch_z, batch_labels = batch_z.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                outputs = classifier(batch_z)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
        # Eval
        classifier.eval()
        val_loss = 0.0
        with torch.no_grad():
            if cfg.FUSION_METHOD == 'Attention':
                bx_val = torch.tensor(X_trad_val, dtype=torch.float32).to(device)
                by_val = torch.tensor(Y_deep_val, dtype=torch.float32).to(device)
                bl_val = torch.tensor(y_val, dtype=torch.long).to(device)
                outputs_val = classifier(bx_val, by_val)
                loss = criterion(outputs_val, bl_val)
            else:
                batch_z_val = torch.tensor(Z_val, dtype=torch.float32).to(device)
                bl_val = torch.tensor(y_val, dtype=torch.long).to(device)
                outputs_val = classifier(batch_z_val)
                loss = criterion(outputs_val, bl_val)
            val_loss = loss.item()
            
        print(f"  Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_classifier_state = classifier.state_dict().copy()
        else:
            patience_counter += 1
            
        if cfg.USE_EARLY_STOPPING and patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break
            
    classifier.load_state_dict(best_classifier_state)
    
    # Generate predictions & probabilities for validation
    classifier.eval()
    with torch.no_grad():
        if cfg.FUSION_METHOD == 'Attention':
            bx_val = torch.tensor(X_trad_val, dtype=torch.float32).to(device)
            by_val = torch.tensor(Y_deep_val, dtype=torch.float32).to(device)
            logits = classifier(bx_val, by_val)
        else:
            batch_z_val = torch.tensor(Z_val, dtype=torch.float32).to(device)
            logits = classifier(batch_z_val)
            
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
    return preds, probs

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Resolve data root dynamically if not specified
    data_root = args.data_root
    if data_root is None:
        data_root = cfg.get_dataset_path()
        
    print(f"Dataset root directory: {data_root}")
    data_dir, csv_path = download_and_extract_dataset(data_root)
    
    # Initialize main training & validation datasets separately for augmentations
    train_dataset = PhysioNet2017Dataset(
        data_dir=data_dir,
        csv_path=csv_path,
        max_len_sec=args.max_len_sec,
        mode='pad',
        preprocess=True,
        is_train=True
    )
    
    val_dataset = PhysioNet2017Dataset(
        data_dir=data_dir,
        csv_path=csv_path,
        max_len_sec=args.max_len_sec,
        mode='pad',
        preprocess=True,
        is_train=False
    )
    
    labels = np.array([r['label_idx'] for r in val_dataset.records])
    
    # 2. Pre-extract traditional handcrafted features from validation dataset (clean)
    if os.access(data_root, os.W_OK):
        cache_dir = data_root
    else:
        cache_dir = "."
    cache_path = os.path.join(cache_dir, "traditional_features_cache.npz")
    X_trad, _ = preextract_traditional_features(val_dataset, cache_path=cache_path)
    
    # 3. K-Fold Cross Validation Setup
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    
    all_fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- Training Fold {fold+1}/{args.folds} ---")
        
        # Create subsets from augmented and clean datasets respectively
        train_sub = Subset(train_dataset, train_idx)
        val_sub = Subset(val_dataset, val_idx)
        
        # Calculate dynamic class weights for loss function
        train_labels = labels[train_idx]
        class_counts = np.bincount(train_labels, minlength=4)
        total_samples = len(train_labels)
        class_weights = total_samples / (4.0 * class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        print(f"Class counts: {class_counts}, Weights: {class_weights.numpy()}")
        
        # Setup Sampler or Shuffle
        if cfg.USE_BALANCED_SAMPLER:
            class_counts_safe = np.clip(class_counts, a_min=1, a_max=None)
            class_weights_sampler = 1.0 / class_counts_safe
            sample_weights = class_weights_sampler[train_labels]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_sub, batch_size=args.batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True)
            
        val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False)
        
        # Extract traditional features for current splits
        X_trad_train = X_trad[train_idx]
        X_trad_val = X_trad[val_idx]
        
        # 4. Train Deep Learning Feature Extractor
        print("Training Deep Feature Extractor...")
        deep_model = ResNetGRUFeatureExtractor(gru_hidden=args.gru_hidden).to(device)
        deep_model = train_deep_model(
            deep_model, train_loader, val_loader, class_weights, 
            epochs=args.deep_epochs, device=device
        )
        
        # Extract Deep Features (Y)
        train_loader_ordered = DataLoader(train_sub, batch_size=args.batch_size, shuffle=False)
        Y_deep_train, y_train = extract_deep_features(deep_model, train_loader_ordered, device)
        Y_deep_val, y_val = extract_deep_features(deep_model, val_loader, device)
        
        # Double check alignment
        assert np.array_equal(y_train, train_labels), "Label alignment mismatch in train split!"
        
        # 5. Fit Feature Fusion and Classifier
        if cfg.FUSION_METHOD == 'DCCA':
            print("Fitting DCCA feature fusion...")
            dcca = DCCAProjection(d_components=args.dcca_dim, eta=args.dcca_eta, reg=1e-4)
            dcca.fit(X_trad_train, Y_deep_train, y_train)
            
            # Project features
            Z_train = dcca.transform(X_trad_train, Y_deep_train)
            Z_val = dcca.transform(X_trad_val, Y_deep_val)
            
            classifier = FusionClassifier(input_dim=Z_train.shape[1], num_classes=4).to(device)
            Z_train_pass, Z_val_pass = Z_train, Z_val
            
        elif cfg.FUSION_METHOD == 'Concatenate':
            print("Using direct Concatenation baseline...")
            Z_train = np.hstack((X_trad_train, Y_deep_train))
            Z_val = np.hstack((X_trad_val, Y_deep_val))
            
            classifier = FusionClassifier(input_dim=Z_train.shape[1], num_classes=4).to(device)
            Z_train_pass, Z_val_pass = Z_train, Z_val
            
        else: # 'Attention'
            print("Using Gated Cross-Attention fusion...")
            trad_dim = X_trad_train.shape[1]
            deep_dim = Y_deep_train.shape[1]
            classifier = AttentionFusionClassifier(trad_dim=trad_dim, deep_dim=deep_dim, num_classes=4).to(device)
            Z_train_pass = (X_trad_train, Y_deep_train)
            Z_val_pass = (X_trad_val, Y_deep_val)
            
        # 6. Train Fusion Classifier
        print("Training Fusion Classifier...")
        preds, probs = train_fusion_classifier(
            classifier, Z_train_pass, y_train, Z_val_pass, y_val, class_weights,
            epochs=args.classifier_epochs, batch_size=args.batch_size, device=device
        )
        
        # 7. Evaluate Performance
        fold_res = calculate_metrics(y_val, preds, probs)
        all_fold_metrics.append(fold_res)
        
        print(f"Fold {fold+1} Results:")
        print(f"  F1 Normal: {fold_res['F1_Normal']:.4f} | Precision: {fold_res['Precision_Normal']:.4f} | Recall: {fold_res['Recall_Normal']:.4f}")
        print(f"  F1 AF:     {fold_res['F1_AF']:.4f} | Precision: {fold_res['Precision_AF']:.4f} | Recall: {fold_res['Recall_AF']:.4f}")
        print(f"  F1 Other:  {fold_res['F1_Other']:.4f} | Precision: {fold_res['Precision_Other']:.4f} | Recall: {fold_res['Recall_Other']:.4f}")
        print(f"  F1 Noise:  {fold_res['F1_Noise']:.4f} | Precision: {fold_res['Precision_Noise']:.4f} | Recall: {fold_res['Recall_Noise']:.4f}")
        print(f"  F1 Overall (CinC): {fold_res['F1_Overall']:.4f} | Accuracy: {fold_res['Accuracy']:.4f}")
        print(f"  AF Sensitivity: {fold_res['Sensitivity_AF']:.4f} | AF Specificity: {fold_res['Specificity_AF']:.4f} | AF ROC-AUC: {fold_res['ROC_AUC_AF']:.4f}")
        print("  Confusion Matrix:")
        print(fold_res['Confusion_Matrix'])
        
        # Save fold checkpoints
        checkpoint_dir = "./models_checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(deep_model.state_dict(), os.path.join(checkpoint_dir, f"fold_{fold+1}_deep.pt"))
        torch.save(classifier.state_dict(), os.path.join(checkpoint_dir, f"fold_{fold+1}_classifier.pt"))
        if cfg.FUSION_METHOD == 'DCCA':
            with open(os.path.join(checkpoint_dir, f"fold_{fold+1}_dcca.pkl"), 'wb') as f:
                pickle.dump(dcca, f)
        
    # --- Print Average Metrics across all Folds ---
    print("\n================ FINAL CROSS-VALIDATION RESULTS ================")
    f1_n_list = [m['F1_Normal'] for m in all_fold_metrics]
    f1_a_list = [m['F1_AF'] for m in all_fold_metrics]
    f1_o_list = [m['F1_Other'] for m in all_fold_metrics]
    f1_p_list = [m['F1_Noise'] for m in all_fold_metrics]
    f1_ov_list = [m['F1_Overall'] for m in all_fold_metrics]
    acc_list = [m['Accuracy'] for m in all_fold_metrics]
    sen_list = [m['Sensitivity_AF'] for m in all_fold_metrics]
    spe_list = [m['Specificity_AF'] for m in all_fold_metrics]
    auc_list = [m['ROC_AUC_AF'] for m in all_fold_metrics]
    
    # Calculate macro confusion matrix
    cm_all = np.sum([m['Confusion_Matrix'] for m in all_fold_metrics], axis=0)
    
    print(f"Average F1 Normal:  {np.mean(f1_n_list):.4f} +/- {np.std(f1_n_list):.4f}")
    print(f"Average F1 AF:      {np.mean(f1_a_list):.4f} +/- {np.std(f1_a_list):.4f}")
    print(f"Average F1 Other:   {np.mean(f1_o_list):.4f} +/- {np.std(f1_o_list):.4f}")
    print(f"Average F1 Noise:   {np.mean(f1_p_list):.4f} +/- {np.std(f1_p_list):.4f}")
    print(f"Average F1 Overall: {np.mean(f1_ov_list):.4f} +/- {np.std(f1_ov_list):.4f}")
    print(f"Average Accuracy:   {np.mean(acc_list):.4f} +/- {np.std(acc_list):.4f}")
    print(f"Average AF Sensitivity: {np.mean(sen_list):.4f}")
    print(f"Average AF Specificity: {np.mean(spe_list):.4f}")
    print(f"Average AF ROC-AUC:     {np.mean(auc_list):.4f} +/- {np.std(auc_list):.4f}")
    print("\nAccumulated Confusion Matrix across all folds:")
    print(cm_all)
    
    # Run Ensemble Evaluation over the entire validation dataset
    print("\n================ RUNNING ENSEMBLE EVALUATION ================ ")
    checkpoint_dir = "./models_checkpoint"
    ensemble_models = []
    ensemble_classifiers = []
    ensemble_dccas = []
    
    for f_idx in range(args.folds):
        m = ResNetGRUFeatureExtractor(gru_hidden=args.gru_hidden).to(device)
        m.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"fold_{f_idx+1}_deep.pt"), map_location=device))
        m.eval()
        ensemble_models.append(m)
        
        if cfg.FUSION_METHOD == 'DCCA':
            clf = FusionClassifier(input_dim=args.dcca_dim * 2, num_classes=4).to(device)
        elif cfg.FUSION_METHOD == 'Concatenate':
            clf = FusionClassifier(input_dim=X_trad.shape[1] + m.out_dim, num_classes=4).to(device)
        else:
            clf = AttentionFusionClassifier(trad_dim=X_trad.shape[1], deep_dim=m.out_dim, num_classes=4).to(device)
            
        clf.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"fold_{f_idx+1}_classifier.pt"), map_location=device))
        clf.eval()
        ensemble_classifiers.append(clf)
        
        if cfg.FUSION_METHOD == 'DCCA':
            with open(os.path.join(checkpoint_dir, f"fold_{f_idx+1}_dcca.pkl"), 'rb') as f:
                dcca_obj = pickle.load(f)
            ensemble_dccas.append(dcca_obj)
            
    # Extract deep features on full dataset for each model
    full_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    Y_deeps = []
    for f_idx in range(args.folds):
        print(f"Extracting deep features for Fold {f_idx+1} model...")
        Y_deep, _ = extract_deep_features(ensemble_models[f_idx], full_loader, device)
        Y_deeps.append(Y_deep)
        
    all_ensemble_probs = []
    N_samples = len(val_dataset)
    
    print("Computing ensemble soft-voting predictions...")
    for i in range(N_samples):
        sample_probs = []
        x_trad_i = X_trad[i:i+1] # [1, trad_dim]
        
        for f_idx in range(args.folds):
            y_deep_i = Y_deeps[f_idx][i:i+1]
            clf = ensemble_classifiers[f_idx]
            
            if cfg.FUSION_METHOD == 'DCCA':
                z_i = ensemble_dccas[f_idx].transform(x_trad_i, y_deep_i)
                z_tensor = torch.tensor(z_i, dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = clf(z_tensor)
            elif cfg.FUSION_METHOD == 'Concatenate':
                z_i = np.hstack((x_trad_i, y_deep_i))
                z_tensor = torch.tensor(z_i, dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = clf(z_tensor)
            else:
                x_t_tensor = torch.tensor(x_trad_i, dtype=torch.float32).to(device)
                y_d_tensor = torch.tensor(y_deep_i, dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = clf(x_t_tensor, y_d_tensor)
                    
            probs_i = torch.softmax(logits, dim=1).cpu().numpy()[0]
            sample_probs.append(probs_i)
            
        avg_probs = np.mean(sample_probs, axis=0)
        all_ensemble_probs.append(avg_probs)
        
    all_ensemble_probs = np.array(all_ensemble_probs)
    ensemble_preds = np.argmax(all_ensemble_probs, axis=1)
    
    ens_metrics = calculate_metrics(labels, ensemble_preds, all_ensemble_probs)
    
    print("\n================ ENSEMBLE EVALUATION RESULTS ================")
    print(f"Ensemble F1 Normal:  {ens_metrics['F1_Normal']:.4f} | Precision: {ens_metrics['Precision_Normal']:.4f} | Recall: {ens_metrics['Recall_Normal']:.4f}")
    print(f"Ensemble F1 AF:      {ens_metrics['F1_AF']:.4f} | Precision: {ens_metrics['Precision_AF']:.4f} | Recall: {ens_metrics['Recall_AF']:.4f}")
    print(f"Ensemble F1 Other:   {ens_metrics['F1_Other']:.4f} | Precision: {ens_metrics['Precision_Other']:.4f} | Recall: {ens_metrics['Recall_Other']:.4f}")
    print(f"Ensemble F1 Noise:   {ens_metrics['F1_Noise']:.4f} | Precision: {ens_metrics['Precision_Noise']:.4f} | Recall: {ens_metrics['Recall_Noise']:.4f}")
    print(f"Ensemble F1 Overall: {ens_metrics['F1_Overall']:.4f} | Accuracy: {ens_metrics['Accuracy']:.4f}")
    print(f"Ensemble AF Sensitivity: {ens_metrics['Sensitivity_AF']:.4f} | AF Specificity: {ens_metrics['Specificity_AF']:.4f} | AF ROC-AUC: {ens_metrics['ROC_AUC_AF']:.4f}")
    print("Ensemble Confusion Matrix:")
    print(ens_metrics['Confusion_Matrix'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AF Detection with DCCA Feature Fusion")
    parser.add_argument("--data_root", type=str, default=None, help="Root directory for dataset")
    parser.add_argument("--folds", type=int, default=cfg.FOLDS, help="Number of cross-validation folds")
    parser.add_argument("--max_len_sec", type=int, default=cfg.MAX_LEN_SEC, help="Max signal duration in seconds")
    parser.add_argument("--gru_hidden", type=int, default=cfg.GRU_HIDDEN, help="GRU hidden units")
    parser.add_argument("--dcca_dim", type=int, default=cfg.DCCA_DIM, help="Dimensions after DCCA projection")
    parser.add_argument("--dcca_eta", type=float, default=cfg.DCCA_ETA, help="DCCA interclass weight parameter")
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--deep_epochs", type=int, default=cfg.DEEP_EPOCHS, help="Epochs to train Deep model")
    parser.add_argument("--classifier_epochs", type=int, default=cfg.CLASSIFIER_EPOCHS, help="Epochs to train Fusion Classifier")
    parser.add_argument("--seed", type=int, default=cfg.SEED, help="Random seed")
    
    args = parser.parse_args()
    main(args)
