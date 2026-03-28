"""
PGD Attack Evaluation Script.

Evaluates model robustness against PGD (Projected Gradient Descent) attack
and generates detailed visualizations:
    - Confusion matrices
    - ROC curves
    - Score distributions
    - Reliability diagrams (calibration)
    - Robustness curve (Accuracy/AUC vs Epsilon)
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------- Paths ----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(ROOT, "dataset", "processed")
IMAGE_DIR = os.path.join(DATA_DIR, "images_all")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

MODEL_PATH = os.path.join(ROOT, "models", "resnet18_chestxray.pth")

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

EPS_LIST = [0, 0.25/255, 0.5/255, 0.75/255, 1/255, 2/255]

# PGD parameters
PGD_STEPS = 10
PGD_ALPHA = 1/255


def eps_label(eps: float) -> str:
    """Format eps for plot titles."""
    if eps == 0:
        return "0"
    val = eps * 255
    known = [0.25, 0.5, 0.75, 1.0, 2.0]
    for k in known:
        if abs(val - k) < 1e-9:
            return f"{k:g}/255"
    return f"{eps:.6f}"


# ---------------- Dataset ----------------
class NIHBinaryDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.image_dir, row["Image Index"])).convert("RGB")
        y = int(row["binary_label"])
        if self.transform:
            img = self.transform(img)
        return img, y


# ---------------- Model ----------------
class NormalizeWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.backbone(x)


def build_model():
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, 2)
    model = NormalizeWrapper(backbone).to(device)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def make_loader(max_samples=None):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    ds = NIHBinaryDataset(TEST_CSV, IMAGE_DIR, tf)
    if max_samples is not None and max_samples < len(ds):
        ds = Subset(ds, list(range(max_samples)))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return dl


# ---------------- PGD Attack ----------------
def pgd_attack(model, x, y, eps, steps=PGD_STEPS, alpha=None):
    """PGD (Projected Gradient Descent) attack."""
    if eps == 0:
        return x
    
    if alpha is None:
        alpha = min(eps / 4, PGD_ALPHA)
    
    x_orig = x.detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0, 1)
    
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = nn.CrossEntropyLoss()(logits, y)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x_orig, -eps, eps)
        x_adv = torch.clamp(x_orig + delta, 0, 1).detach()
    
    return x_adv


# ---------------- ECE ----------------
def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0
    for i in range(n_bins):
        mask = binids == i
        if np.sum(mask) > 0:
            acc = np.mean(y_true[mask] == (y_prob[mask] > 0.5))
            conf = np.mean(y_prob[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / len(y_prob)
    return ece


# ---------------- Evaluation ----------------
def evaluate(model, loader, eps, steps=PGD_STEPS):
    """Evaluate model under PGD attack at given epsilon."""
    y_true, y_score, y_pred = [], [], []
    
    for x, y in tqdm(loader, desc=f"PGD eps={eps_label(eps)}", leave=False):
        x, y = x.to(device), y.to(device)
        
        with torch.enable_grad():
            x_adv = pgd_attack(model, x, y, eps, steps=steps)
        
        with torch.no_grad():
            logits = model(x_adv)
            probs = torch.softmax(logits, dim=1)
            score = probs[:, 1]
            pred = probs.argmax(1)
        
        y_true.append(y.cpu())
        y_score.append(score.cpu())
        y_pred.append(pred.cpu())
    
    y_true = torch.cat(y_true).numpy()
    y_score = torch.cat(y_score).numpy()
    y_pred = torch.cat(y_pred).numpy()
    
    acc = (y_pred == y_true).mean()
    auc = roc_auc_score(y_true, y_score)
    ece = compute_ece(y_true, y_score)
    
    return acc, auc, ece, y_true, y_pred, y_score


# ---------------- Plots ----------------
def plot_confusion(y_true, y_pred, eps):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(f"PGD Confusion Matrix (ε={eps_label(eps)})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"pgd_confusion_eps{eps}.png"),
                dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_roc(y_true, y_score, eps):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"PGD ROC Curve (ε={eps_label(eps)})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"pgd_roc_eps{eps}.png"),
                dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_score_dist(y_true, y_score, eps):
    """Plot score distribution."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.kdeplot(pos, label="Positive (Disease)", ax=ax)
    sns.kdeplot(neg, label="Negative (Healthy)", ax=ax)
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlabel("Prediction Score")
    ax.set_ylabel("Density")
    ax.set_title(f"PGD Score Distribution (ε={eps_label(eps)})")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"pgd_score_eps{eps}.png"),
                dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_reliability(y_true, y_score, eps):
    """Plot reliability diagram (calibration curve)."""
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker='o', label="Model")
    ax.plot([0, 1], [0, 1], '--', color='gray', label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"PGD Reliability Diagram (ε={eps_label(eps)})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"pgd_reliability_eps{eps}.png"),
                dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


# ---------------- Main ----------------
def main(max_samples=None, eps_list=None, steps=PGD_STEPS):
    """Run PGD evaluation."""
    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")
    print(f"PGD Steps: {steps}")
    
    model = build_model()
    loader = make_loader(max_samples=max_samples)
    
    if eps_list is None:
        eps_list = EPS_LIST
    
    print(f"Epsilon values: {[eps_label(e) for e in eps_list]}")
    
    results = []
    acc_list, auc_list = [], []
    
    for eps in eps_list:
        acc, auc, ece, y_true, y_pred, y_score = evaluate(model, loader, eps, steps=steps)
        print(f"eps={eps_label(eps):>8s}  acc={acc:.4f}  auc={auc:.4f}  ece={ece:.4f}")
        
        acc_list.append(acc)
        auc_list.append(auc)
        results.append({"eps": eps, "accuracy": acc, "auc": auc, "ece": ece})
        
        # Generate detailed plots for selected epsilon values
        if eps in [0, eps_list[len(eps_list)//2], eps_list[-1]]:
            plot_confusion(y_true, y_pred, eps)
            plot_roc(y_true, y_score, eps)
            plot_score_dist(y_true, y_score, eps)
            plot_reliability(y_true, y_score, eps)
    
    # Robustness curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps_list, acc_list, 'o-', label="Accuracy", linewidth=2, markersize=6)
    ax.plot(eps_list, auc_list, 's-', label="AUC", linewidth=2, markersize=6)
    ax.set_xlabel("Epsilon (ε)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"PGD Robustness Curve (steps={steps})", fontsize=12, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "pgd_robustness_curve.png"),
                dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    
    # Save results to JSON
    with open(os.path.join(RESULTS_DIR, "pgd_eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGD Attack Evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Limit number of test samples for faster runs")
    parser.add_argument("--eps_subset", nargs="*", type=float, default=None,
                       help="Epsilon values to evaluate (default: predefined EPS_LIST)")
    parser.add_argument("--steps", type=int, default=PGD_STEPS,
                       help=f"Number of PGD steps (default: {PGD_STEPS})")
    args = parser.parse_args()
    main(max_samples=args.max_samples, eps_list=args.eps_subset, steps=args.steps)
