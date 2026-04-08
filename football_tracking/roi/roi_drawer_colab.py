import cv2
import matplotlib.pyplot as plt


ROI_NAMES = [
    "center_pitch",
    "left_penalty_box",
    "right_penalty_box",
    "left_goal",
    "right_goal",
]


def annotate_rois_colab(image_path):
    # load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title("Reference Frame")
    plt.axis("off")
    plt.show()

    print("\n👉 Enter coordinates as list of (x,y) points")
    print("Example: [(100,200),(300,200),(300,400),(100,400)]\n")

    rois = {}

    for name in ROI_NAMES:
        user_input = input(f"Enter ROI for {name}: ")
        rois[name] = eval(user_input)

    return rois