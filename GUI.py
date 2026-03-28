import cv2

frame = cv2.imread("debug_init_frame.jpg")
display = cv2.resize(frame, (1280, 720))

cv2.namedWindow("Select Ball", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select Ball", 1280, 720)  # 👈 force size

bbox = cv2.selectROI("Select Ball", frame, False)
cv2.destroyAllWindows()


scale_x = frame.shape[1] / display.shape[1]
scale_y = frame.shape[0] / display.shape[0]

x, y, w, h = bbox

bbox = (
    int(x * scale_x),
    int(y * scale_y),
    int(w * scale_x),
    int(h * scale_y)
)
print("Selected:", bbox)