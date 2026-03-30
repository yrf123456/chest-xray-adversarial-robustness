import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


BASE_DIR = "E:/project_root/project_root/dataset/processed"
CSV_PATH = os.path.join(BASE_DIR, "train.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "images_all")



class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = row["Image Index"]
        label = int(row["binary_label"])

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])



if __name__ == "__main__":
    dataset = NIHChestXrayDataset(
        csv_file=CSV_PATH,
        image_dir=IMAGE_DIR,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    images, labels = next(iter(dataloader))

    print("Images shape:", images.shape)
    print("Labels:", labels)
