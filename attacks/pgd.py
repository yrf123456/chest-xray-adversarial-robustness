import os
import json
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
OUT_CSV = os.path.join(RESULTS_DIR, "robust_pgd_metrics.csv")
OUT_PNG = os.path.join(RESULTS_DIR, "robust_pgd_acc_auc_vs_eps.png")
OUT_JSON = os.path.join(RESULTS_DIR, "robust_pgd_summary.json")


# ---------------- Hyperparams ----------------
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# epsilon in pixel space [0,1]
EPS_LIST = [0.0, 0.25/255, 0.5/255, 0.75/255, 1/255, 2/255]

# PGD params
PGD_STEPS = 10
PGD_ALPHA_BASE = 1 / 255   # 实际使用时会 min(alpha, eps)


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
        img = Image.open(
            os.path.join(self.image_dir, row["Image Index"])
        ).convert("RGB")
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


# ---------------- Metrics helpers ----------------
def safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0


def confusion_from_pred(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    return tn, fp, fn, tp


# ---------------- PGD attack ----------------
def pgd_attack(model, x, y, eps, steps, alpha_base):
    """
    PGD (L_inf)
    x: [B,3,H,W] in [0,1]
    """
    if eps == 0.0:
        return x

    x_orig = x.detach()

    # random start
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    alpha = min(alpha_base, eps)

    for _ in range(steps):
        x_adv.requires_grad_(True)

        logits = model(x_adv)
        loss = nn.CrossEntropyLoss()(logits, y)

        model.zero_grad(set_to_none=True)
        loss.backward()

        grad = x_adv.grad.detach()

        x_adv = x_adv + alpha * torch.sign(grad)

        # projection
        delta = torch.clamp(x_adv - x_orig, -eps, eps)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()

    return x_adv


@torch.no_grad()
def forward_probs(model, x):
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    return probs


def eval_one_eps_pgd(model, loader, eps):
    y_true_all = []
    y_score_all = []
    y_pred_all = []

    correct, total = 0, 0

    for x, y in tqdm(loader, desc=f"PGD eps={eps:.5f}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.enable_grad():
            x_adv = pgd_attack(
                model, x, y,
                eps=eps,
                steps=PGD_STEPS,
                alpha_base=PGD_ALPHA_BASE
            )

        probs = forward_probs(model, x_adv)
        score_pos = probs[:, 1]
        pred = probs.argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

        y_true_all.append(y.cpu())
        y_score_all.append(score_pos.cpu())
        y_pred_all.append(pred.cpu())

    y_true = torch.cat(y_true_all).numpy()
    y_score = torch.cat(y_score_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()

    acc = correct / total
    auc = float(roc_auc_score(y_true, y_score))

    tn, fp, fn, tp = confusion_from_pred(y_true, y_pred)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    bal_acc = 0.5 * (recall + specificity)

    return {
        "eps": eps,
        "acc": acc,
        "auc": auc,
        "precision": precision,
        "recall_tpr": recall,
        "specificity_tnr": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
        "balanced_acc": bal_acc,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }


def plot_curves(df, out_png):
    plt.figure()
    plt.plot(df["eps"], df["acc"], marker="o", label="Accuracy")
    plt.plot(df["eps"], df["auc"], marker="o", label="AUC")
    plt.xlabel("epsilon (L_inf, pixel space [0,1])")
    plt.ylabel("score")
    plt.title("PGD Robustness on NIH Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")
    print(f"Test CSV: {TEST_CSV}")
    print("EPS:", EPS_LIST)
    print(f"PGD steps={PGD_STEPS}, alpha_base={PGD_ALPHA_BASE}")

    model = build_model()
    ds, dl = make_loader()
    print(f"Test size: {len(ds)}")

    rows = []
    for eps in EPS_LIST:
        rows.append(eval_one_eps_pgd(model, dl, eps))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    plot_curves(df, OUT_PNG)

    summary = {
        "model_path": MODEL_PATH,
        "test_csv": TEST_CSV,
        "eps_list": EPS_LIST,
        "pgd_steps": PGD_STEPS,
        "pgd_alpha_base": PGD_ALPHA_BASE,
        "outputs": {"csv": OUT_CSV, "png": OUT_PNG}
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(" ", OUT_CSV)
    print(" ", OUT_PNG)
    print(" ", OUT_JSON)
    print("\nPreview:\n", df[["eps", "acc", "auc", "recall_tpr", "specificity_tnr"]])


if __name__ == "__main__":
    main()
