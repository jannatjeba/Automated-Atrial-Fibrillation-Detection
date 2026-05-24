import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def calculate_metrics(y_true, y_pred):
    """
    Calculates classification metrics for PhysioNet CinC 2017 Challenge.
    Classes: 0: Normal (N), 1: AF (A), 2: Other (O), 3: Noise (~)
    
    Metrics returned:
        F1n: F1 score for Normal class
        F1a: F1 score for AF class
        F1o: F1 score for Other class
        F1p: F1 score for Noise class
        Foverall: Average of F1n, F1a, and F1o
        accuracy: Total accuracy
        sensitivity: Sensitivity for AF class (macro or per-class)
        specificity: Specificity for AF class (macro or per-class)
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    # Calculate F1 for each class
    # F1 = 2 * TP / (2*TP + FP + FN)
    f1_scores = []
    for c in range(4):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1_scores.append(f1)
        
    f1_n, f1_a, f1_o, f1_p = f1_scores
    
    # PhysioNet 2017 Challenge Overall F1 is average of Normal, AF, and Other
    f1_overall = (f1_n + f1_a + f1_o) / 3.0
    
    # Total Accuracy
    total_samples = np.sum(cm)
    accuracy = np.trace(cm) / total_samples if total_samples > 0 else 0.0
    
    # Sensitivity (Sen) and Specificity (Spe) for AF (Class 1) vs all others
    # AF is positive, Others are negative
    tp_af = cm[1, 1]
    fn_af = np.sum(cm[1, :]) - tp_af
    fp_af = np.sum(cm[:, 1]) - tp_af
    tn_af = total_samples - (tp_af + fn_af + fp_af)
    
    sensitivity = tp_af / (tp_af + fn_af + 1e-8)
    specificity = tn_af / (tn_af + fp_af + 1e-8)
    
    return {
        'F1_Normal': f1_n,
        'F1_AF': f1_a,
        'F1_Other': f1_o,
        'F1_Noise': f1_p,
        'F1_Overall': f1_overall,
        'Accuracy': accuracy,
        'Sensitivity_AF': sensitivity,
        'Specificity_AF': specificity,
        'Confusion_Matrix': cm
    }

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt
