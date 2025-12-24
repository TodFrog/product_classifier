#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Test Tool (Safe Mode)
카메라 하드웨어 설정을 건드리지 않고 단순히 카메라 화면만 확인

- 카메라 하드웨어 설정 변경 없음
- CLAHE 후처리만 테스트 가능
"""

import cv2
import numpy as np

# Camera IDs
CAM_TOP = 0
CAM_SIDE = 2


def apply_clahe(frame, clip_limit=2.0):
    """Apply CLAHE preprocessing"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_channel_clahe = clahe.apply(l_channel)
    lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def main():
    print("=" * 60)
    print("Camera Test Tool (Safe Mode)")
    print("=" * 60)
    print("카메라 하드웨어 설정을 건드리지 않습니다.")
    print("CLAHE 후처리만 테스트합니다.")
    print()
    print("Controls:")
    print("  c - CLAHE ON/OFF 토글")
    print("  q - 종료")
    print("=" * 60)

    # Open cameras - NO settings modification
    cap_top = cv2.VideoCapture(CAM_TOP, cv2.CAP_V4L2)
    cap_side = cv2.VideoCapture(CAM_SIDE, cv2.CAP_V4L2)

    if not cap_top.isOpened():
        print(f"[ERROR] Top camera (ID: {CAM_TOP}) 열기 실패")
        return

    if not cap_side.isOpened():
        print(f"[ERROR] Side camera (ID: {CAM_SIDE}) 열기 실패")
        cap_top.release()
        return

    # Set resolution only - DO NOT touch exposure settings
    for cap in [cap_top, cap_side]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"[OK] Top Camera (ID {CAM_TOP}) 열림")
    print(f"[OK] Side Camera (ID {CAM_SIDE}) 열림")

    # Create window
    cv2.namedWindow('Camera Test', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Test', 1280, 480)

    clahe_enabled = False

    while True:
        # Read frames
        ret_top, frame_top = cap_top.read()
        ret_side, frame_side = cap_side.read()

        if not ret_top or not ret_side:
            print("[ERROR] 프레임 읽기 실패")
            break

        # Apply CLAHE if enabled
        if clahe_enabled:
            frame_top = apply_clahe(frame_top)
            frame_side = apply_clahe(frame_side)

        # Add labels
        cv2.putText(frame_top, f"TOP (CAM {CAM_TOP})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame_side, f"SIDE (CAM {CAM_SIDE})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show CLAHE status
        clahe_text = "CLAHE: ON" if clahe_enabled else "CLAHE: OFF"
        cv2.putText(frame_top, clahe_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame_side, clahe_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Combine frames
        combined = np.hstack([frame_top, frame_side])

        # Show
        cv2.imshow('Camera Test', combined)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            clahe_enabled = not clahe_enabled
            print(f"[INFO] CLAHE: {'ON' if clahe_enabled else 'OFF'}")

    # Cleanup
    cap_top.release()
    cap_side.release()
    cv2.destroyAllWindows()

    print("\n[INFO] 종료됨")


if __name__ == '__main__':
    main()
