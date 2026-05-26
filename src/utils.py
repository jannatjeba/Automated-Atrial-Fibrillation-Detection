import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def calculate_metrics(y_true, y_pred, y_probs=None):
    """
    Calculates classification metrics for PhysioNet CinC 2017 Challenge.
    Classes: 0: Normal (N), 1: AF (A), 2: Other (O), 3: Noise (~)
    
    Metrics returned:
        F1_Normal, F1_AF, F1_Other, F1_Noise
        Precision_Normal, Precision_AF, Precision_Other, Precision_Noise
        Recall_Normal, Recall_AF, Recall_Other, Recall_Noise
        F1_Overall: Average of F1_Normal, F1_AF, and F1_Other
        Accuracy: Total accuracy
        Sensitivity_AF: Sensitivity for AF class
        Specificity_AF: Specificity for AF class
        ROC_AUC_AF: ROC-AUC for AF class
        Confusion_Matrix: raw confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    # Calculate F1, Precision, and Recall for each class
    f1_scores = []
    precisions = []
    recalls = []
    for c in range(4):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        
        denom_f1 = 2 * tp + fp + fn
        f1 = (2 * tp / denom_f1) if denom_f1 > 0 else 0.0
        f1_scores.append(f1)
        
        denom_prec = tp + fp
        prec = (tp / denom_prec) if denom_prec > 0 else 0.0
        precisions.append(prec)
        
        denom_recall = tp + fn
        rec = (tp / denom_recall) if denom_recall > 0 else 0.0
        recalls.append(rec)
        
    f1_n, f1_a, f1_o, f1_p = f1_scores
    prec_n, prec_a, prec_o, prec_p = precisions
    rec_n, rec_a, rec_o, rec_p = recalls
    
    # PhysioNet 2017 Challenge Overall F1 is average of Normal, AF, and Other
    f1_overall = (f1_n + f1_a + f1_o) / 3.0
    
    # Total Accuracy
    total_samples = np.sum(cm)
    accuracy = np.trace(cm) / total_samples if total_samples > 0 else 0.0
    
    # Sensitivity (Sen) and Specificity (Spe) for AF (Class 1) vs all others
    tp_af = cm[1, 1]
    fn_af = np.sum(cm[1, :]) - tp_af
    fp_af = np.sum(cm[:, 1]) - tp_af
    tn_af = total_samples - (tp_af + fn_af + fp_af)
    
    sensitivity = tp_af / (tp_af + fn_af + 1e-8)
    specificity = tn_af / (tn_af + fp_af + 1e-8)
    
    # ROC-AUC for AF class
    roc_auc_af = 0.0
    if y_probs is not None:
        try:
            from sklearn.metrics import roc_auc_score
            y_true_af = (np.array(y_true) == 1).astype(int)
            y_probs_af = np.array(y_probs)[:, 1]
            if len(np.unique(y_true_af)) > 1:
                roc_auc_af = roc_auc_score(y_true_af, y_probs_af)
        except Exception:
            pass
            
    return {
        'F1_Normal': f1_n,
        'F1_AF': f1_a,
        'F1_Other': f1_o,
        'F1_Noise': f1_p,
        'Precision_Normal': prec_n,
        'Precision_AF': prec_a,
        'Precision_Other': prec_o,
        'Precision_Noise': prec_p,
        'Recall_Normal': rec_n,
        'Recall_AF': rec_a,
        'Recall_Other': rec_o,
        'Recall_Noise': rec_p,
        'F1_Overall': f1_overall,
        'Accuracy': accuracy,
        'Sensitivity_AF': sensitivity,
        'Specificity_AF': specificity,
        'ROC_AUC_AF': roc_auc_af,
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
