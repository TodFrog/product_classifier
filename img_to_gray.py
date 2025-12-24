import cv2
import numpy as np
from ultralytics import YOLO
import os

# =======================
# ⚙️ 설정
# =======================
MODEL_PATH = "runs/detect/train/weights/best.pt"
IMAGE_PATH = "test.jpg"          # 단일 이미지
# IMAGE_PATH = "test_images/"    # 폴더도 가능

CLIP_LIMIT = 2.0
GRID_SIZE = (8, 8)
SHARPEN_KERNEL = np.array([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
])

clahe = cv2.createCLAHE(
    clipLimit=CLIP_LIMIT,
    tileGridSize=GRID_SIZE
)

def gray_sharpen_transform(img):
    """
    BGR image → gray + CLAHE + sharpen → 3ch BGR
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    sharpened = cv2.filter2D(enhanced, -1, SHARPEN_KERNEL)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

# =======================
#  YOLO Inference
# =======================
def run_inference(img_path):
    model = YOLO(MODEL_PATH)

    #  단일 이미지
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"cannot read: {img_path}")

        img = gray_sharpen_transform(img)
        results = model(img)

        results[0].show()  # 시각화

    #  폴더
    elif os.path.isdir(img_path):
        for name in os.listdir(img_path):
            if not name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            full_path = os.path.join(img_path, name)
            img = cv2.imread(full_path)
            if img is None:
                continue

            img = gray_sharpen_transform(img)
            results = model(img)

            results[0].show()

    else:
        raise ValueError("IMAGE_PATH should be file/folder.")

if __name__ == "__main__":
    run_inference(IMAGE_PATH)
