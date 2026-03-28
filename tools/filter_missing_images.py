import os
import pandas as pd

ROOT = r"E:\project_root\project_root"
PROCESSED_DIR = os.path.join(ROOT, "dataset", "processed")
IMAGE_DIR = os.path.join(PROCESSED_DIR, "images_all")


def filter_csv(csv_name):
    csv_path = os.path.join(PROCESSED_DIR, csv_name)
    df = pd.read_csv(csv_path)

    # ① 一次性读取 images_all 中所有文件名
    existing_files = set(os.listdir(IMAGE_DIR))

    # ② 用内存集合判断（极快）
    mask = df["Image Index"].isin(existing_files)

    removed = (~mask).sum()
    print(f"{csv_name}: remove {removed} missing images")

    df_clean = df[mask].reset_index(drop=True)
    df_clean.to_csv(csv_path, index=False)


if __name__ == "__main__":
    filter_csv("train.csv")
    filter_csv("val.csv")
    filter_csv("test.csv")
