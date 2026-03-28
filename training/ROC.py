import os
import json
import numpy as np
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn

from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "dataset", "processed")
IMAGE_DIR = os.path.join(DATA_DIR, "images_all")
MODEL_PATH = os.path.join(ROOT, "models", "resnet18_chestxray.pth")

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 输出文件
METRICS_JSON = os.path.join(RESULTS_DIR, "test_metrics.json")
ROC_POINTS_CSV = os.path.join(RESULTS_DIR, "roc_points_test.csv")
ROC_PNG = os.path.join(RESULTS_DIR, "ROC.png")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        if self.transform:
            img = self.transform(img)
        return img, int(row["binary_label"])


class NormalizeWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return self.backbone(x)


def safe_div(a, b):
    return float(a) / float(b) if b != 0 else 0.0


def main():
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    test_csv = os.path.join(DATA_DIR, "test.csv")
    test_set = NIHBinaryDataset(test_csv, IMAGE_DIR, tf)

    loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, 2)
    model = NormalizeWrapper(backbone).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # 收集：真实标签、预测概率、预测类别
    y_true = []
    y_score = []  # 正类(1)概率，用于 AUC/ROC
    y_pred = []

    correct, total = 0, 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating (acc + auc + roc)"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            score_pos = probs[:, 1]  # P(class=1)

            pred = torch.argmax(logits, dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

            y_true.append(y.detach().cpu())
            y_score.append(score_pos.detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true).numpy().astype(int)
    y_score = torch.cat(y_score).numpy().astype(float)
    y_pred = torch.cat(y_pred).numpy().astype(int)

    # Accuracy
    acc = correct / total

    # AUC + ROC points
    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)

    # 保存 ROC 点
    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds
    })
    roc_df.to_csv(ROC_POINTS_CSV, index=False)

    # 混淆矩阵（基于 argmax 等价于阈值0.5左右，但更严格来说是二分类logits比较）
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())

    # 各指标（以“1=有病”为正类）
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)                 # Sensitivity / TPR
    specificity = safe_div(tn, tn + fp)            # TNR
    fpr_at_pred = safe_div(fp, fp + tn)
    fnr_at_pred = safe_div(fn, fn + tp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    balanced_acc = 0.5 * (recall + specificity)

    # 画 ROC 并保存
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve (Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_PNG, dpi=200)
    plt.close()

    metrics = {
        "model_path": MODEL_PATH,
        "test_csv": test_csv,
        "test_size": int(len(test_set)),
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "accuracy": round(float(acc), 6),
        "auc": round(float(auc), 6),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "metrics": {
            "precision": round(float(precision), 6),
            "recall_sensitivity_tpr": round(float(recall), 6),
            "specificity_tnr": round(float(specificity), 6),
            "fpr": round(float(fpr_at_pred), 6),
            "fnr": round(float(fnr_at_pred), 6),
            "f1": round(float(f1), 6),
            "balanced_accuracy": round(float(balanced_acc), 6),
        },
        "outputs": {
            "roc_points_csv": ROC_POINTS_CSV,
            "roc_png": ROC_PNG
        }
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== Test Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"CM: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Saved metrics to: {METRICS_JSON}")
    print(f"Saved ROC points to: {ROC_POINTS_CSV}")
    print(f"Saved ROC figure to: {ROC_PNG}")


if __name__ == "__main__":
    main()
