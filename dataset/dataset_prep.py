import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = "E:\\project_root\\project_root\\dataset\\raw"
OUTPUT_DIR = "E:\\project_root\\project_root\\dataset\\processed"

IMAGES_ALL_DIR = os.path.join(OUTPUT_DIR, "images_all")
os.makedirs(IMAGES_ALL_DIR, exist_ok=True)


def merge_images():
    """
    Merge images_001 ~ images_012 into processed/images_all,
    automatically handle nested 'images/' folder.
    """
    print("Merging images...")

    for folder in sorted(os.listdir(RAW_DIR)):
        fp = os.path.join(RAW_DIR, folder)
        if os.path.isdir(fp) and folder.startswith("images_"):

            print(f"  Processing {folder} ...")

            
            inner = os.path.join(fp, "images")
            src_dir = inner if os.path.isdir(inner) else fp

            
            for img in os.listdir(src_dir):
                src_path = os.path.join(src_dir, img)
                if os.path.isfile(src_path):
                    shutil.copy(src_path, os.path.join(IMAGES_ALL_DIR, img))

    print("Image merge complete.\n")



def build_label_csv():
    """
    Parse Data_Entry_2017.csv and build binary labels
    """
    print("Building labels...")

    df = pd.read_csv(os.path.join(RAW_DIR, "Data_Entry_2017.csv"))

    def to_binary(label):
        return 0 if label == "No Finding" else 1

    df["binary_label"] = df["Finding Labels"].apply(to_binary)

    df.to_csv(os.path.join(OUTPUT_DIR, "all_labels.csv"), index=False)
    print("Saved: all_labels.csv\n")


def split_train_val_test():
    """
    Use NIH official train_val_list.txt and test_list.txt to split dataset
    """
    print("Splitting train / val / test...")

    df = pd.read_csv(os.path.join(OUTPUT_DIR, "all_labels.csv"))

    with open(os.path.join(RAW_DIR, "test_list.txt")) as fp:
        test_files = [line.strip() for line in fp]

    with open(os.path.join(RAW_DIR, "train_val_list.txt")) as fp:
        train_val_files = [line.strip() for line in fp]

    test_df = df[df["Image Index"].isin(test_files)]
    train_val_df = df[df["Image Index"].isin(train_val_files)]

    train_df, val_df = train_test_split(train_val_df,
                                        test_size=0.1,
                                        random_state=42)

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print("Saved: train.csv, val.csv, test.csv\n")


if __name__ == "__main__":
    merge_images()
    build_label_csv()
    split_train_val_test()

    print("Dataset preparation complete!")

