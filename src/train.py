import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from dataset import download_and_extract_dataset, PhysioNet2017Dataset
from features import extract_features
from models import ResNetGRUFeatureExtractor, DCCAProjection, FusionClassifier
from utils import calculate_metrics

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    
    # We iterate over the dataframe records to load full signals for traditional features
    for idx in tqdm(range(len(dataset))):
        record_info = dataset.records[idx]
        record_name = record_info['record']
        label = record_info['label_idx']
        
        # Load and preprocess full signal
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
    Trains the ResNet + GRU model with a temporary classification head.
    """
    # Temporary classification head for pretraining the deep feature extractor
    classifier_head = nn.Linear(model.out_dim, 4).to(device)
    
    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier_head.parameters()), 
        lr=cfg.DEEP_LR, weight_decay=cfg.DEEP_WD
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        classifier_head.train()
        train_loss = 0.0
        
        for signals, masks, labels, _ in train_loader:
            signals, masks, labels = signals.to(device), masks.to(device), labels.to(device)
            
            optimizer.zero_grad()
            deep_features = model(signals, masks)
            outputs = classifier_head(deep_features)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * signals.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        classifier_head.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for signals, masks, labels, _ in val_loader:
                signals, masks, labels = signals.to(device), masks.to(device), labels.to(device)
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
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                'model': model.state_dict(),
                'classifier_head': classifier_head.state_dict()
            }
            
    # Restore best weights
    model.load_state_dict(best_model_state['model'])
    return model

def extract_deep_features(model, dataloader, device):
    """
    Extracts deep learning features using the trained ResNet-GRU extractor.
    """
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for signals, masks, labels, _ in dataloader:
            signals, masks = signals.to(device), masks.to(device)
            deep_feat = model(signals, masks)
            features_list.append(deep_feat.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)

def train_fusion_classifier(classifier, Z_train, y_train, Z_val, y_val, class_weights, epochs=40, batch_size=64, device='cpu'):
    """
    Trains the Fusion Classifier on DCCA projected features.
    """
    optimizer = optim.Adam(classifier.parameters(), lr=cfg.CLASSIFIER_LR, weight_decay=cfg.CLASSIFIER_WD)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Create simple numpy dataloaders
    train_data = torch.utils.data.TensorDataset(
        torch.tensor(Z_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    best_classifier_state = None
    
    for epoch in range(epochs):
        classifier.train()
        for batch_z, batch_y in train_loader:
            batch_z, batch_y = batch_z.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = classifier(batch_z)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        # Eval
        classifier.eval()
        val_loss = 0.0
        with torch.no_grad():
            batch_z_val = torch.tensor(Z_val, dtype=torch.float32).to(device)
            batch_y_val = torch.tensor(y_val, dtype=torch.long).to(device)
            outputs_val = classifier(batch_z_val)
            loss = criterion(outputs_val, batch_y_val)
            val_loss = loss.item()
            
        print(f"  Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_classifier_state = classifier.state_dict().copy()
            
    classifier.load_state_dict(best_classifier_state)
    
    # Generate predictions
    classifier.eval()
    with torch.no_grad():
        batch_z_val = torch.tensor(Z_val, dtype=torch.float32).to(device)
        logits = classifier(batch_z_val)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
    return preds

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
    
    # Initialize main dataset
    full_dataset = PhysioNet2017Dataset(
        data_dir=data_dir,
        csv_path=csv_path,
        max_len_sec=args.max_len_sec,
        mode='pad',
        preprocess=True
    )
    
    labels = np.array([r['label_idx'] for r in full_dataset.records])
    
    # 2. Pre-extract traditional handcrafted features
    if os.access(data_root, os.W_OK):
        cache_dir = data_root
    else:
        cache_dir = "."
    cache_path = os.path.join(cache_dir, "traditional_features_cache.npz")
    X_trad, _ = preextract_traditional_features(full_dataset, cache_path=cache_path)
    
    # 3. K-Fold Cross Validation Setup
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    
    all_fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- Training Fold {fold+1}/{args.folds} ---")
        
        # Create subsets
        train_sub = Subset(full_dataset, train_idx)
        val_sub = Subset(full_dataset, val_idx)
        
        # Calculate dynamic class weights for deep network training
        train_labels = labels[train_idx]
        class_counts = np.bincount(train_labels, minlength=4)
        total_samples = len(train_labels)
        # Class weighting formula: total_samples / (num_classes * count)
        class_weights = total_samples / (4.0 * class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        print(f"Class counts: {class_counts}, Weights: {class_weights.numpy()}")
        
        train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False)
        
        # Extract features for current splits
        X_trad_train = X_trad[train_idx]
        X_trad_val = X_trad[val_idx]
        
        # 4. Train Deep Learning Feature Extractor (ResNet + GRU)
        print("Training Deep Feature Extractor (ResNet + GRU)...")
        deep_model = ResNetGRUFeatureExtractor(gru_hidden=args.gru_hidden).to(device)
        deep_model = train_deep_model(
            deep_model, train_loader, val_loader, class_weights, 
            epochs=args.deep_epochs, device=device
        )
        
        # Extract Deep Features (Y)
        # We need ordered loader for feature extraction matching indexes
        train_loader_ordered = DataLoader(train_sub, batch_size=args.batch_size, shuffle=False)
        Y_deep_train, y_train = extract_deep_features(deep_model, train_loader_ordered, device)
        Y_deep_val, y_val = extract_deep_features(deep_model, val_loader, device)
        
        # Double check alignment
        assert np.array_equal(y_train, train_labels), "Label alignment mismatch in train split!"
        
        # 5. Fit DCCA projection on current fold strictly to avoid leakage
        print("Fitting DCCA feature fusion...")
        dcca = DCCAProjection(d_components=args.dcca_dim, eta=args.dcca_eta, reg=1e-4)
        dcca.fit(X_trad_train, Y_deep_train, y_train)
        
        # Project features
        Z_train = dcca.transform(X_trad_train, Y_deep_train)
        Z_val = dcca.transform(X_trad_val, Y_deep_val)
        
        # 6. Train Fusion Classifier
        print("Training Fusion Classifier...")
        classifier = FusionClassifier(input_dim=Z_train.shape[1], num_classes=4).to(device)
        preds = train_fusion_classifier(
            classifier, Z_train, y_train, Z_val, y_val, class_weights,
            epochs=args.classifier_epochs, batch_size=args.batch_size, device=device
        )
        
        # 7. Evaluate Performance
        fold_res = calculate_metrics(y_val, preds)
        all_fold_metrics.append(fold_res)
        
        print(f"Fold {fold+1} Results:")
        print(f"  F1 Normal: {fold_res['F1_Normal']:.4f}")
        print(f"  F1 AF:     {fold_res['F1_AF']:.4f}")
        print(f"  F1 Other:  {fold_res['F1_Other']:.4f}")
        print(f"  F1 Overall (CinC): {fold_res['F1_Overall']:.4f}")
        print(f"  Accuracy:  {fold_res['Accuracy']:.4f}")
        
    # --- Print Average Metrics across all Folds ---
    print("\n================ FINAL CROSS-VALIDATION RESULTS ================")
    f1_n_list = [m['F1_Normal'] for m in all_fold_metrics]
    f1_a_list = [m['F1_AF'] for m in all_fold_metrics]
    f1_o_list = [m['F1_Other'] for m in all_fold_metrics]
    f1_ov_list = [m['F1_Overall'] for m in all_fold_metrics]
    acc_list = [m['Accuracy'] for m in all_fold_metrics]
    sen_list = [m['Sensitivity_AF'] for m in all_fold_metrics]
    spe_list = [m['Specificity_AF'] for m in all_fold_metrics]
    
    print(f"Average F1 Normal: {np.mean(f1_n_list):.4f} +/- {np.std(f1_n_list):.4f}")
    print(f"Average F1 AF:     {np.mean(f1_a_list):.4f} +/- {np.std(f1_a_list):.4f}")
    print(f"Average F1 Other:  {np.mean(f1_o_list):.4f} +/- {np.std(f1_o_list):.4f}")
    print(f"Average F1 Overall:{np.mean(f1_ov_list):.4f} +/- {np.std(f1_ov_list):.4f}")
    print(f"Average Accuracy:  {np.mean(acc_list):.4f} +/- {np.std(acc_list):.4f}")
    print(f"Average AF Sensitivity: {np.mean(sen_list):.4f}")
    print(f"Average AF Specificity: {np.mean(spe_list):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AF Detection with DCCA Feature Fusion")
    parser.add_argument("--data_root", type=str, default=None, help="Root directory for dataset (default: resolved from config / path file)")
    parser.add_argument("--folds", type=int, default=cfg.FOLDS, help="Number of cross-validation folds")
    parser.add_argument("--max_len_sec", type=int, default=cfg.MAX_LEN_SEC, help="Max signal duration in seconds")
    parser.add_argument("--gru_hidden", type=int, default=cfg.GRU_HIDDEN, help="GRU hidden units")
    parser.add_argument("--dcca_dim", type=int, default=cfg.DCCA_DIM, help="Dimensions after DCCA projection")
    parser.add_argument("--dcca_eta", type=float, default=cfg.DCCA_ETA, help="DCCA interclass weight parameter")
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--deep_epochs", type=int, default=cfg.DEEP_EPOCHS, help="Epochs to pretrain ResNet-GRU")
    parser.add_argument("--classifier_epochs", type=int, default=cfg.CLASSIFIER_EPOCHS, help="Epochs to train Fusion Classifier")
    parser.add_argument("--seed", type=int, default=cfg.SEED, help="Random seed")
    
    args = parser.parse_args()
    
    # Run the main pipeline
    main(args)
