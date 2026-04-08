import shutil
import os

def merge_data(src_img, src_lbl, dst_img_train, dst_lbl_train):
    for f in os.listdir(src_img):
        print(f)
        # rename file prefix in the same folder

        new_name = f.replace("None_", "BM_")
        shutil.copy(
            os.path.join(src_img, f),
            os.path.join(dst_img_train, new_name)
        )
        # shutil.copy(
        #     os.path.join(src_img, f),
        #     os.path.join(dst_img_train, f)
        # )

    for f in os.listdir(src_lbl):
        new_name = f.replace("None_", "BM_")
        shutil.copy(
            os.path.join(src_lbl, f),
            os.path.join(dst_lbl_train, new_name)
        )
        # shutil.copy(
        #     os.path.join(src_lbl, f),
        #     os.path.join(dst_lbl_train, f)
        # )

merge_data(
    "../../new_video_frames/March29_3/Images",
    "../../new_video_frames/March29_3/labels",
    "../../dataset/images/",
    "../../dataset/labels/"
)