"""
Comprehensive Robustness Evaluation Script.

Compares model robustness against FGSM and PGD attacks on the same plot.
Generates:
    - FGSM vs PGD comparison curves (Accuracy and AUC vs Epsilon)
    - Summary report (CSV and JSON)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt


# ---------------- Paths ----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "dataset", "processed")
IMAGE_DIR = os.path.join(DATA_DIR, "images_all")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
MODEL_PATH = os.path.join(ROOT, "models", "resnet18_chestxray.pth")

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- Hyperparams ----------------
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS_LIST = [0.0, 0.25/255, 0.5/255, 0.75/255, 1/255, 2/255]


def eps_label(eps: float) -> str:
    """Format eps for display."""
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
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, 
                   num_workers=NUM_WORKERS, pin_memory=True)
    return ds, dl


# ---------------- Attack Functions ----------------
def fgsm_attack(model, x, y, eps):
    """FGSM Attack."""
    if eps == 0.0:
        return x
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits, y)
    model.zero_grad()
    loss.backward()
    grad = x_adv.grad.detach()
    x_adv = x_adv + eps * torch.sign(grad)
    return torch.clamp(x_adv, 0.0, 1.0).detach()


def pgd_attack(model, x, y, eps, steps=10, alpha=None):
    """PGD Attack."""
    if eps == 0.0:
        return x
    if alpha is None:
        alpha = eps / 4

    x_orig = x.detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = nn.CrossEntropyLoss()(logits, y)
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x_orig, -eps, eps)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()

    return x_adv


# ---------------- Evaluation ----------------
def evaluate_attack(model, loader, attack_fn, attack_name, eps):
    """Evaluate model robustness against a specific attack."""
    y_true_all, y_score_all, y_pred_all = [], [], []
    
    for x, y in tqdm(loader, desc=f"{attack_name.upper()} ε={eps_label(eps)}", leave=False):
        x, y = x.to(device), y.to(device)
        
        with torch.enable_grad():
            x_adv = attack_fn(model, x, y, eps)
        
        with torch.no_grad():
            logits = model(x_adv)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
        
        y_true_all.extend(y.cpu().tolist())
        y_score_all.extend(probs[:, 1].cpu().tolist())
        y_pred_all.extend(pred.cpu().tolist())
    
    y_true = np.array(y_true_all)
    y_score = np.array(y_score_all)
    y_pred = np.array(y_pred_all)
    
    acc = (y_pred == y_true).mean()
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = 0.5
    
    return {"attack": attack_name, "eps": eps, "accuracy": acc, "auc": auc}


def run_comparison(model, loader, eps_list=None):
    """Run FGSM vs PGD comparison."""
    if eps_list is None:
        eps_list = EPS_LIST
    
    results = []
    
    print("\n=== FGSM Evaluation ===")
    for eps in eps_list:
        metrics = evaluate_attack(model, loader, fgsm_attack, "fgsm", eps)
        results.append(metrics)
        print(f"  ε={eps_label(eps):>8s}  acc={metrics['accuracy']:.4f}  auc={metrics['auc']:.4f}")
    
    print("\n=== PGD Evaluation ===")
    for eps in eps_list:
        metrics = evaluate_attack(model, loader, pgd_attack, "pgd", eps)
        results.append(metrics)
        print(f"  ε={eps_label(eps):>8s}  acc={metrics['accuracy']:.4f}  auc={metrics['auc']:.4f}")
    
    return pd.DataFrame(results)


# ---------------- Visualization ----------------
def plot_comparison_curves(df, save_path=None):
    """Plot FGSM vs PGD comparison curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {"fgsm": "#2196F3", "pgd": "#F44336"}
    markers = {"fgsm": "o", "pgd": "s"}
    
    for attack in ["fgsm", "pgd"]:
        attack_df = df[df["attack"] == attack]
        axes[0].plot(attack_df["eps"], attack_df["accuracy"], 
                     marker=markers[attack], color=colors[attack],
                     label=attack.upper(), linewidth=2, markersize=7)
        axes[1].plot(attack_df["eps"], attack_df["auc"], 
                     marker=markers[attack], color=colors[attack],
                     label=attack.upper(), linewidth=2, markersize=7)
    
    axes[0].set_xlabel("Epsilon (ε)", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].set_title("Accuracy vs Perturbation", fontsize=12, fontweight='bold')
    axes[0].legend(loc="lower left")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)
    
    axes[1].set_xlabel("Epsilon (ε)", fontsize=11)
    axes[1].set_ylabel("AUC", fontsize=11)
    axes[1].set_title("AUC vs Perturbation", fontsize=12, fontweight='bold')
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)
    
    fig.suptitle("FGSM vs PGD Robustness Comparison", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved: {save_path}")
    
    plt.close(fig)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="FGSM vs PGD Robustness Comparison")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of test samples (for faster evaluation)")
    parser.add_argument("--eps", nargs="+", type=float, default=None,
                       help="Epsilon values (default: predefined list)")
    args = parser.parse_args()
    
    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")
    
    model = build_model()
    ds, dl = make_loader(max_samples=args.max_samples)
    print(f"Test size: {len(ds)}")
    
    eps_list = args.eps if args.eps else EPS_LIST
    print(f"Epsilon values: {[eps_label(e) for e in eps_list]}")
    
    # Run comparison
    df = run_comparison(model, dl, eps_list)
    
    # Save results
    csv_path = os.path.join(RESULTS_DIR, "robustness_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")
    
    # Plot comparison curves
    plot_path = os.path.join(RESULTS_DIR, "robustness_comparison.png")
    plot_comparison_curves(df, plot_path)
    
    # Save summary JSON
    summary = {
        "fgsm": {
            "min_accuracy": float(df[df["attack"] == "fgsm"]["accuracy"].min()),
            "max_accuracy": float(df[df["attack"] == "fgsm"]["accuracy"].max()),
            "min_auc": float(df[df["attack"] == "fgsm"]["auc"].min()),
            "max_auc": float(df[df["attack"] == "fgsm"]["auc"].max()),
        },
        "pgd": {
            "min_accuracy": float(df[df["attack"] == "pgd"]["accuracy"].min()),
            "max_accuracy": float(df[df["attack"] == "pgd"]["accuracy"].max()),
            "min_auc": float(df[df["attack"] == "pgd"]["auc"].min()),
            "max_auc": float(df[df["attack"] == "pgd"]["auc"].max()),
        }
    }
    
    json_path = os.path.join(RESULTS_DIR, "robustness_comparison.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON saved: {json_path}")
    
    print("\n=== Summary ===")
    print(f"FGSM: Accuracy {summary['fgsm']['min_accuracy']:.4f} - {summary['fgsm']['max_accuracy']:.4f}")
    print(f"PGD:  Accuracy {summary['pgd']['min_accuracy']:.4f} - {summary['pgd']['max_accuracy']:.4f}")


if __name__ == "__main__":
    main()
