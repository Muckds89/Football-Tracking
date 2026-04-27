import os
import random
import shutil
from pathlib import Path


def split_dataset(base_dir="../../datasets", val_ratio=0.2, test_ratio=0.1):
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")

    train_img_dir = os.path.join(images_dir, "train")
    val_img_dir = os.path.join(images_dir, "val")
    test_img_dir = os.path.join(images_dir, "test")

    train_lbl_dir = os.path.join(labels_dir, "train")
    val_lbl_dir = os.path.join(labels_dir, "val")
    test_lbl_dir = os.path.join(labels_dir, "test")

    for d in [
        train_img_dir, val_img_dir, test_img_dir,
        train_lbl_dir, val_lbl_dir, test_lbl_dir
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Support multiple formats
    images = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    n_total = len(images)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)

    test_files = images[:n_test]
    val_files = images[n_test:n_test + n_val]
    train_files = images[n_test + n_val:]

    def move(files, img_dest, lbl_dest):
        for f in files:
            src_img = os.path.join(images_dir, f)
            src_lbl = os.path.join(labels_dir, Path(f).stem + ".txt")

            dst_img = os.path.join(img_dest, f)
            dst_lbl = os.path.join(lbl_dest, Path(f).stem + ".txt")

            shutil.move(src_img, dst_img)

            # IMPORTANT: create empty label if missing
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)
            else:
                Path(dst_lbl).touch()  # empty file = negative image

    move(train_files, train_img_dir, train_lbl_dir)
    move(val_files, val_img_dir, val_lbl_dir)
    move(test_files, test_img_dir, test_lbl_dir)

    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")
    print(f"Test: {len(test_files)}")


split_dataset()