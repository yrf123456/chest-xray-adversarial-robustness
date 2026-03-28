import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "dataset", "processed")
IMAGE_DIR = os.path.join(DATA_DIR, "images_all")
MODEL_PATH = os.path.join(ROOT, "models", "resnet18_chestxray.pth")

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


def main():
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    test_set = NIHBinaryDataset(
        os.path.join(DATA_DIR, "test.csv"),
        IMAGE_DIR,
        tf
    )

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

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    correct, total = 0, 0
    for x, y in tqdm(loader, desc="Evaluating test accuracy"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"[Test Accuracy] {acc:.4f}")


if __name__ == "__main__":
    main()

