import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO

# --- 0. Grayscale 전처리 설정 ---
USE_GRAYSCALE = True  # True: grayscale 전처리 적용, False: 원본 사용

CLIP_LIMIT = 2.0
GRID_SIZE = (8, 8)
SHARPEN_KERNEL = np.array([
    [0, -1,  0],
    [-1, 5, -1],
    [0, -1,  0]
], dtype=np.float32)

clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=GRID_SIZE)

def gray_sharpen_transform(img):
    """BGR image → gray + CLAHE + sharpen → 3ch BGR"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    sharpened = cv2.filter2D(enhanced, -1, SHARPEN_KERNEL)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

# --- 1. 테스트 환경 설정 (환경에 맞게 수정) ---

# [A. 🚀 Jetson Orin 테스트 시]
# (테스트하려는 모델의 주석을 해제하세요)
# MODEL_PATH = "ckpt/yolov8n_ep600_p70.pt"       # 1. PyTorch 원본
# MODEL_PATH = "yolov8n_ep600_p70.onnx"      # 2. ONNX (FP16)
# MODEL_PATH = "/home/chai/Desktop/classifier/yolo_train/ckpt/1126_13sub_aug_best.engine"  # 3. TensorRT (FP16) - Best Performance

# Grayscale 모델 (권장)
MODEL_PATH = "/home/chai/Desktop/classifier/product_classifier/src/yolo_models/1222_v8n_img480_best_aug_segment_gray.pt"

# [B. 🟢 Intel N97 테스트 시]
# (테스트하려는 모델의 주석을 해제하세요)
# MODEL_PATH = "ckpt/yolov8n_ep600_p70.pt"                  # 1. PyTorch 원본 (CPU/GPU)
# MODEL_PATH = "yolov8n_ep600_p70_openvino_model/"     # 2. OpenVINO (FP16) - Best Performance

# 자동 디바이스 감지
if torch.cuda.is_available():
    DEVICE = '0'  # CUDA GPU
else:
    DEVICE = 'cpu'  # CPU fallback

# ★★★ 2대의 카메라 설정 ★★★
CAMERA_INDICES = [0, 2]
# ----------------------------------------------------

print(f"Loading model: {MODEL_PATH}")
print(f"Using device: {DEVICE}")
print(f"Grayscale preprocessing: {'Enabled' if USE_GRAYSCALE else 'Disabled'}")

# 2. 모델 1회 로드
try:
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 3. 여러 카메라 및 FPS 카운터 초기화
caps = []
p_times = {} # 각 카메라별로 p_time을 저장할 딕셔너리
window_names = []

for index in CAMERA_INDICES:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {index}.")
    else:
        caps.append(cap)
        p_times[index] = 0  # FPS 계산용 시간 초기화
        window_name = f"YOLOv8 Test - Camera {index}"
        window_names.append(window_name)
        cv2.namedWindow(window_name)
        print(f"Opened Camera {index}")

if not caps:
    print("No cameras could be opened. Exiting.")
    exit()

print("Starting webcam feeds... (Press 'q' to quit)")

while True:
    
    for i, cap in enumerate(caps):
        cam_index = CAMERA_INDICES[i]
        window_name = window_names[i]
        
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame from camera {cam_index}")
            continue

        # 4. 전처리 (grayscale + CLAHE + sharpen)
        if USE_GRAYSCALE:
            processed_frame = gray_sharpen_transform(frame)
        else:
            processed_frame = frame

        # 5. 추론
        results = model.predict(processed_frame, device=DEVICE, verbose=False)

        # 6. FPS 계산 (카메라별)
        c_time = time.time()
        current_p_time = p_times[cam_index]
        if current_p_time > 0:
            fps = 1 / (c_time - current_p_time)
        else:
            fps = 0  # 첫 프레임
        p_times[cam_index] = c_time

        # 7. 결과 시각화 (전처리된 프레임에 표시)
        annotated_frame = results[0].plot()

        # 8. 정보 표시 (FPS, 카메라 ID, 모델, 전처리 상태)
        gray_status = "GRAY" if USE_GRAYSCALE else "COLOR"
        cv2.putText(annotated_frame, f"Cam {cam_index} | FPS: {int(fps)} | {gray_status}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Model: {MODEL_PATH.split('/')[-1]}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 9. 개별 창에 표시
        cv2.imshow(window_name, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. 정리
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
print("Test finished.")