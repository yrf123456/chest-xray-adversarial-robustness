import os
import time
import json
from datetime import datetime

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ================== 路径与常量 ==================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "dataset", "processed")
IMAGE_DIR = os.path.join(DATA_DIR, "images_all")

MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "resnet18_chestxray.pth")
SUMMARY_JSON = os.path.join(RESULT_DIR, "train_summary.json")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================== Dataset ==================
class NIHBinaryDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["Image Index"])
        label = int(row["binary_label"])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


# ================== 工具函数 ==================
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class NormalizeWrapper(nn.Module):
    """将 ImageNet mean/std 内嵌进模型"""
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


@torch.no_grad()
def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


# ================== DataLoader ==================
def make_loaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    train_set = NIHBinaryDataset(os.path.join(DATA_DIR, "train.csv"), IMAGE_DIR, train_tf)
    val_set   = NIHBinaryDataset(os.path.join(DATA_DIR, "val.csv"),   IMAGE_DIR, eval_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    return train_set, val_set, train_loader, val_loader


# ================== Model ==================
def build_model():
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, 2)
    return NormalizeWrapper(backbone).to(device)


# ================== Main ==================
def main():
    run_start = time.perf_counter()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")

    train_set, val_set, train_loader, val_loader = make_loaders()
    print(f"Dataset sizes | train={len(train_set)} val={len(val_set)}")
    print(f"Hyperparams | epochs={EPOCHS} batch={BATCH_SIZE} lr={LR}")

    model = build_model()
    total_params, trainable_params = count_params(model)
    print(f"Model params | total={total_params/1e6:.2f}M trainable={trainable_params/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    total_seen = 0
    epoch_times = []

    for ep in range(1, EPOCHS + 1):
        ep_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{EPOCHS}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            seen += y.size(0)
            total_seen += y.size(0)

        val_acc = evaluate_accuracy(model, val_loader)
        ep_time = time.perf_counter() - ep_start
        epoch_times.append(ep_time)

        print(
            f"[Val] ep={ep:02d} acc={val_acc:.4f} "
            f"loss={running_loss/len(train_loader):.4f} "
            f"time={ep_time:.1f}s | throughput≈{seen/ep_time:.1f} img/s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("Saved best model:", MODEL_PATH)

    total_train_time = time.perf_counter() - run_start
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    summary = {
        "start_time": start_dt,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "best_val_acc": round(best_val_acc, 4),
        "total_images_seen": total_seen,
        "total_train_time_sec": round(total_train_time, 2),
        "avg_epoch_time_sec": round(avg_epoch_time, 2),
        "epoch_times_sec": [round(t, 2) for t in epoch_times],
        "model_path": MODEL_PATH
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training finished.")
    print("Summary saved to:", SUMMARY_JSON)


if __name__ == "__main__":
    main()
