# AI Smart Vending Machine

Jetson Nano 기반 AI 스마트 자판기 시스템으로, 로드셀(무게 센서)과 듀얼 카메라(YOLOv8)를 융합하여 상품 인식 및 재고 관리를 자동화합니다.

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Smart Vending Machine                  │
├─────────────────────────────────────────────────────────────┤
│  [Top Camera]          [Side Camera]         [Load Cell]    │
│      │                      │                     │         │
│      ▼                      ▼                     ▼         │
│  Hand Detection      Product Detection      Weight Sensing  │
│  (Real-time)         (Motion Filtering)     (Kalman Filter) │
│      │                      │                     │         │
│      └──────────────────────┼─────────────────────┘         │
│                             ▼                               │
│                    ┌─────────────────┐                      │
│                    │  Sensor Fusion  │                      │
│                    │  Vision(CLASS)  │                      │
│                    │  Weight(COUNT)  │                      │
│                    └─────────────────┘                      │
│                             │                               │
│                             ▼                               │
│                    ┌─────────────────┐                      │
│                    │   Tkinter UI    │                      │
│                    │  Status Display │                      │
│                    └─────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

## 핵심 기능

### 1. Sensor Fusion (센서 융합)
- **Vision → CLASS**: YOLOv8로 상품 종류 식별 (14개 클래스)
- **Weight → COUNT**: 로드셀로 수량 계산
- **정확도**: 12회 테스트 중 11회 정확 (91.7%)

### 2. Motion-Based Filtering (움직임 기반 필터링)
- 프레임별 bounding box 좌표 추적
- 평균 이동량 계산 (Euclidean distance)
- 정적 배경 상품 제외, 움직이는 상품만 후보군 등록
- 가려져서 감지 안 되는 프레임 자동 제외

### 3. Real-time Frame Capture
- 무게 변화 감지 후 3초간 (90 frames @ 30fps) 캡처
- 사용자가 상품을 집는 순간을 정확히 포착

### 4. Event Queue System
- 연속 이벤트 순차 처리
- Cancellation Detection (가져갔다 돌려놓기 감지)
- Event Timeout (10초 후 자동 만료)

### 5. 1D Kalman Filter
- 로드셀 노이즈 제거
- 안정적인 무게 측정

## 디렉토리 구조

```
product_classifier/
├── test_fusion_mvp_v2.py      # 메인 시스템 (Fusion + UI)
├── test_loadcell_phase1.py    # 로드셀 단독 테스트
├── test_vision_phase2.py      # 비전 단독 테스트
├── config.yaml                 # 시스템 설정
├── 13subset_label.json         # 상품 라벨 (14개 클래스)
├── requirements.txt            # 의존성 패키지
├── src/
│   ├── core/                   # 핵심 로직
│   │   ├── fusion_engine.py
│   │   ├── object_classifier.py
│   │   ├── state_machine.py
│   │   └── inventory.py
│   ├── filters/                # 신호 처리
│   │   └── kalman_filter.py
│   ├── utils/                  # 유틸리티
│   │   ├── loadcell_driver.py
│   │   └── logger.py
│   └── yolo_models/            # YOLO 모델 파일
│       └── *.engine            # TensorRT 엔진
└── debug_frames/               # 디버그 프레임 (런타임 생성)
```

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

주요 패키지:
- ultralytics (YOLOv8)
- opencv-python
- pyserial
- numpy
- pyyaml
- tkinter (GUI)

### 2. YOLO 모델 다운로드

YOLO 모델 파일(.engine)은 용량 문제로 Git에 포함되지 않습니다.
모델 파일을 `src/yolo_models/` 디렉토리에 배치해주세요.

```bash
# 모델 파일 위치
src/yolo_models/your_model.engine
```

> **Note**: `.engine` 파일은 TensorRT로 최적화된 Jetson 전용 파일입니다.
> 다른 플랫폼에서는 `.pt` 파일을 사용하고 config.yaml의 model_path를 수정하세요.

### 3. 하드웨어 연결

- **Top Camera**: USB 카메라 (ID: 0) - Hand detection
- **Side Camera**: USB 카메라 (ID: 2) - Product detection
- **Load Cell**: Serial 연결 (`/dev/ttyUSB0`)

### 4. 실행

```bash
# 메인 시스템 (Fusion + UI)
python3 test_fusion_mvp_v2.py

# 로드셀 단독 테스트
python3 test_loadcell_phase1.py

# 비전 단독 테스트
python3 test_vision_phase2.py
```

## UI 기능

```
┌────────────────────────────────────────┐
│                                        │
│              READY                     │  ← 상태 표시
│                                        │
│        REMOVED: chickenmayo × 1        │  ← 결과 표시
│                                        │
├────────────────────────────────────────┤
│  ⌨️ Keyboard Controls                  │
│  [R] Restocking  [Z] Manual Zero  [Q] Quit │
├────────────────────────────────────────┤
│  Current Weight: 2431.0g               │
│  Total Transactions: 5                 │
│  Mode: NORMAL                          │
└────────────────────────────────────────┘
```

### 상태 표시

| 상태 | 색상 | 설명 |
|------|------|------|
| READY | 초록 | 대기 중 |
| PROCESSING | 주황 | 판단 중 |
| DETECTED | 파랑 | 결과 표시 |
| ERROR | 빨강 | 오류 발생 |
| RESTOCKING | 보라 | 재고 보충 모드 |

### 키보드 컨트롤

- `R`: Restocking 모드 토글
- `Z`: 수동 영점 조정
- `Q`: 프로그램 종료

## 설정 (config.yaml)

```yaml
# 하드웨어
hardware:
  top_camera_id: 0
  side_camera_id: 2
  loadcell_port: '/dev/ttyUSB0'

# Kalman Filter
kalman:
  process_noise: 0.5
  measurement_noise: 1.0
  dead_zone: 3.0

# 이벤트 감지
event_detection:
  weight_change_threshold: 15.0
  settling_time: 0.5
  variance_threshold: 0.5

# Fusion 설정
fusion:
  inference_frames: 90      # 3초 @ 30fps
  save_debug_frames: true
  tolerance_percent: 0.10   # 무게 오차 허용 ±10%

# 제품 데이터베이스
products:
  1:
    name: 'chickenmayo'
    weight: 92.0
    is_product: true
```

## State Machine

```
IDLE → EVENT_TRIGGER → INTERACTION → VERIFICATION → IDLE
  │                                        │
  │         (weight returned)              │
  └────────── CANCELLED ◄──────────────────┘
```

- **IDLE**: 무게 모니터링, 대기 상태
- **EVENT_TRIGGER**: 무게 변화 감지
- **INTERACTION**: 실시간 프레임 캡처 (3초)
- **VERIFICATION**: 센서 융합 및 결과 검증

## 문제 해결

### 로드셀 연결 실패
```
Failed to connect to load cell
```
→ `/dev/ttyUSB0` 포트 확인 또는 config.yaml에서 수정

### 카메라 열기 실패
```
Failed to open camera
```
→ camera_id 확인 (0, 2 등)

### 무게 감지 안됨
- `weight_change_threshold` 값 조정
- Kalman filter 파라미터 튜닝

### 정적 배경 상품 오탐
- Motion filtering이 자동으로 처리
- 필요시 `motion_threshold` 조정 (기본값: 10.0 px/frame)

## 로그 파일

- `fusion_mvp_v2_log.txt`: 트랜잭션 로그
- `debug_frames/`: 디버그 프레임 (save_debug_frames: true 시)

## 라이선스

MIT License

## 개발 환경

- **Hardware**: NVIDIA Jetson Nano
- **OS**: Ubuntu 20.04
- **Python**: 3.10+
- **YOLO**: YOLOv8n (TensorRT optimized)
