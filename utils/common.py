import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def safe_div(a, b, default=0.0):
    """Safe division to avoid division by zero."""
    return float(a) / float(b) if b != 0 else default


def save_json(data, filepath):
    """Save dictionary to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath):
    """Load JSON file to dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


class NormalizeWrapper(nn.Module):
    """Wrapper to embed ImageNet normalization into the model."""
    
    def __init__(self, backbone, mean=None, std=None):
        super().__init__()
        self.backbone = backbone
        
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        mean = torch.tensor(mean).view(1, 3, 1, 1)
        std = torch.tensor(std).view(1, 3, 1, 1)
        
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
    
    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return self.backbone(x)


def build_resnet18(num_classes=2, pretrained=False):
    """Build ResNet-18 model for binary classification."""
    weights = "IMAGENET1K_V1" if pretrained else None
    backbone = models.resnet18(weights=weights)
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    return backbone


def build_model(num_classes=2, pretrained=False, device=None):
    """Build wrapped model with normalization layer."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    backbone = build_resnet18(num_classes, pretrained)
    model = NormalizeWrapper(backbone)
    return model.to(device)


def load_model(model_path, num_classes=2, device=None):
    """Load trained model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_model(num_classes, pretrained=False, device=device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def confusion_from_predictions(y_true, y_pred):
    """
    Compute confusion matrix values from predictions.
    
    Returns:
        tn, fp, fn, tp
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    
    return tn, fp, fn, tp


def compute_metrics(y_true, y_pred, y_score=None):
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_score: Prediction scores for positive class (optional, for AUC)
    
    Returns:
        dict with metrics
    """
    from sklearn.metrics import roc_auc_score
    
    tn, fp, fn, tp = confusion_from_predictions(y_true, y_pred)
    
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)  # Sensitivity / TPR
    specificity = safe_div(tn, tn + fp)  # TNR
    f1 = safe_div(2 * precision * recall, precision + recall)
    balanced_acc = 0.5 * (recall + specificity)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "fpr": fpr,
        "fnr": fnr,
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    }
    
    if y_score is not None:
        try:
            auc = float(roc_auc_score(y_true, y_score))
            metrics["auc"] = auc
        except ValueError:
            metrics["auc"] = None
    
    return metrics


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name=""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
