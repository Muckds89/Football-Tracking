import os
import random
import shutil
from pathlib import Path

def split_dataset(base_dir="../../dataset", val_ratio=0.2):
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")

    train_img_dir = os.path.join(images_dir, "train") 
    val_img_dir = os.path.join(images_dir, "val")

    train_lbl_dir = os.path.join(labels_dir, "train")
    val_lbl_dir = os.path.join(labels_dir, "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

    random.shuffle(images)

    split_idx = int(len(images) * (1 - val_ratio))
    train_files = images[:split_idx]
    val_files = images[split_idx:]

    def move(files, img_dest, lbl_dest):
        for f in files:
            src_img = os.path.join(images_dir, f)
            src_lbl = os.path.join(labels_dir, f.replace(".jpg", ".txt"))

            shutil.move(src_img, os.path.join(img_dest, f))
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, os.path.join(lbl_dest, f.replace(".jpg", ".txt")))

    move(train_files, train_img_dir, train_lbl_dir)
    move(val_files, val_img_dir, val_lbl_dir)

    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

split_dataset()