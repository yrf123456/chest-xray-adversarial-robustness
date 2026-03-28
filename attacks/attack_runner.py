"""
Attack Runner - Unified interface for running different adversarial attacks.

Supports:
    - FGSM (Fast Gradient Sign Method)
    - PGD (Projected Gradient Descent)
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
from torch.utils.data import Dataset, DataLoader
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
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
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


def make_loader():
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    ds = NIHBinaryDataset(TEST_CSV, IMAGE_DIR, tf)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    return ds, dl


# ---------------- Attack Functions ----------------
def fgsm_attack(model, x, y, eps):
    """FGSM Attack."""
    if eps == 0.0:
        return x

    x_adv = x.detach().clone()
    x_adv.requires_grad_(True)

    logits = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()

    grad = x_adv.grad.detach()
    x_adv = x_adv + eps * torch.sign(grad)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    return x_adv


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

        model.zero_grad(set_to_none=True)
        loss.backward()

        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha * torch.sign(grad)

        delta = torch.clamp(x_adv - x_orig, -eps, eps)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()

    return x_adv


# ---------------- Attack Runner ----------------
class AttackRunner:
    """Unified interface for running adversarial attacks."""
    
    SUPPORTED_ATTACKS = ["fgsm", "pgd"]
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def attack(self, x, y, attack_type, **kwargs):
        """
        Run specified attack on input batch.
        
        Args:
            x: Input images [B, C, H, W]
            y: True labels [B]
            attack_type: Attack type ("fgsm", "pgd")
            **kwargs: Attack-specific parameters
        
        Returns:
            x_adv: Adversarial examples
        """
        x = x.to(self.device)
        y = y.to(self.device)
        
        if attack_type == "fgsm":
            eps = kwargs.get("eps", 8/255)
            return fgsm_attack(self.model, x, y, eps)
        
        elif attack_type == "pgd":
            eps = kwargs.get("eps", 8/255)
            steps = kwargs.get("steps", 10)
            alpha = kwargs.get("alpha", None)
            return pgd_attack(self.model, x, y, eps, steps, alpha)
        
        else:
            raise ValueError(f"Unknown attack type: {attack_type}. "
                           f"Supported: {self.SUPPORTED_ATTACKS}")
    
    def evaluate(self, loader, attack_type, **kwargs):
        """
        Evaluate model robustness under attack.
        
        Returns:
            dict with accuracy, AUC, and other metrics
        """
        y_true_all = []
        y_score_all = []
        y_pred_all = []
        correct, total = 0, 0
        
        desc = f"{attack_type.upper()}"
        if "eps" in kwargs:
            desc += f" eps={kwargs['eps']:.5f}"
        
        for x, y in tqdm(loader, desc=desc, leave=False):
            x = x.to(self.device)
            y = y.to(self.device)
            
            with torch.enable_grad():
                x_adv = self.attack(x, y, attack_type, **kwargs)
            
            with torch.no_grad():
                logits = self.model(x_adv)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1)
            
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            y_true_all.append(y.cpu())
            y_score_all.append(probs[:, 1].cpu())
            y_pred_all.append(pred.cpu())
        
        y_true = torch.cat(y_true_all).numpy()
        y_score = torch.cat(y_score_all).numpy()
        y_pred = torch.cat(y_pred_all).numpy()
        
        acc = correct / total
        auc = float(roc_auc_score(y_true, y_score))
        
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        
        return {
            "attack_type": attack_type,
            "params": kwargs,
            "accuracy": acc,
            "auc": auc,
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
        }
    
    def compare_attacks(self, loader, eps_list=None):
        """
        Compare different attacks at various epsilon values.
        
        Returns:
            DataFrame with results
        """
        if eps_list is None:
            eps_list = [0.0, 1/255, 2/255, 4/255, 8/255]
        
        results = []
        
        for eps in eps_list:
            fgsm_result = self.evaluate(loader, "fgsm", eps=eps)
            fgsm_result["eps"] = eps
            results.append(fgsm_result)
            
            pgd_result = self.evaluate(loader, "pgd", eps=eps, steps=10)
            pgd_result["eps"] = eps
            results.append(pgd_result)
        
        return pd.DataFrame(results)


def run_all_attacks(eps_list=None, save_results=True):
    """Run all attack evaluations and save results."""
    if eps_list is None:
        eps_list = [0.0, 0.25/255, 0.5/255, 0.75/255, 1/255, 2/255]
    
    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")
    
    model = build_model()
    ds, dl = make_loader()
    print(f"Test size: {len(ds)}")
    
    runner = AttackRunner(model, device)
    
    all_results = []
    
    print("\n=== FGSM Attack ===")
    for eps in eps_list:
        result = runner.evaluate(dl, "fgsm", eps=eps)
        result["eps"] = eps
        all_results.append(result)
        print(f"eps={eps:.6f}: acc={result['accuracy']:.4f}, auc={result['auc']:.4f}")
    
    print("\n=== PGD Attack ===")
    for eps in eps_list:
        result = runner.evaluate(dl, "pgd", eps=eps, steps=10)
        result["eps"] = eps
        all_results.append(result)
        print(f"eps={eps:.6f}: acc={result['accuracy']:.4f}, auc={result['auc']:.4f}")
    
    if save_results:
        df = pd.DataFrame(all_results)
        out_csv = os.path.join(RESULTS_DIR, "all_attacks_comparison.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nResults saved to: {out_csv}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Attack Runner")
    parser.add_argument("--attack", type=str, default="fgsm",
                       choices=["fgsm", "pgd", "all"],
                       help="Attack type to run")
    parser.add_argument("--eps", type=float, default=8/255,
                       help="Perturbation budget (epsilon)")
    parser.add_argument("--steps", type=int, default=10,
                       help="Number of attack steps (for PGD)")
    args = parser.parse_args()
    
    print(f"Device: {device}")
    
    model = build_model()
    ds, dl = make_loader()
    
    runner = AttackRunner(model, device)
    
    if args.attack == "all":
        run_all_attacks()
    else:
        kwargs = {"eps": args.eps}
        if args.attack == "pgd":
            kwargs["steps"] = args.steps
        
        result = runner.evaluate(dl, args.attack, **kwargs)
        print(f"\nResults for {args.attack.upper()}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  AUC: {result['auc']:.4f}")
        print(f"  Confusion Matrix: {result['confusion_matrix']}")


if __name__ == "__main__":
    main()
