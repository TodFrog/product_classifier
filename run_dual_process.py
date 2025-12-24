#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual Process Launcher

두 개의 독립 프로세스를 동시에 실행:
- Process 1: 무게 감지 + 이미지 캡처 (process_weight_capture.py)
- Process 2: YOLO 분석 (process_yolo_analysis.py)

사용법:
    python run_dual_process.py

또는 각각 별도 터미널에서:
    Terminal 1: python process_weight_capture.py
    Terminal 2: python process_yolo_analysis.py
"""

import subprocess
import sys
import os
import signal
import time

# 현재 디렉토리
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 프로세스 스크립트
PROCESS1_SCRIPT = os.path.join(SCRIPT_DIR, "process_weight_capture.py")
PROCESS2_SCRIPT = os.path.join(SCRIPT_DIR, "process_yolo_analysis.py")

processes = []


def signal_handler(signum, frame):
    """Handle Ctrl+C"""
    print("\n[Launcher] Shutting down processes...")
    for proc in processes:
        if proc.poll() is None:  # Still running
            proc.terminate()
    sys.exit(0)


def main():
    global processes

    print("=" * 60)
    print("Dual Process Vending Machine System")
    print("=" * 60)
    print()
    print("Starting two independent processes:")
    print("  Process 1: Weight Detection + Image Capture")
    print("  Process 2: YOLO Analysis + UI")
    print()
    print("Press Ctrl+C to stop both processes")
    print("=" * 60)
    print()

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Check if scripts exist
    if not os.path.exists(PROCESS1_SCRIPT):
        print(f"[ERROR] Process 1 script not found: {PROCESS1_SCRIPT}")
        sys.exit(1)

    if not os.path.exists(PROCESS2_SCRIPT):
        print(f"[ERROR] Process 2 script not found: {PROCESS2_SCRIPT}")
        sys.exit(1)

    # Create necessary directories
    os.makedirs("event_queue", exist_ok=True)
    os.makedirs("event_queue/processed", exist_ok=True)
    os.makedirs("captured_events", exist_ok=True)

    try:
        # Start Process 2 first (YOLO - needs time to load model)
        print("[Launcher] Starting Process 2 (YOLO Analysis)...")
        proc2 = subprocess.Popen(
            [sys.executable, PROCESS2_SCRIPT],
            cwd=SCRIPT_DIR
        )
        processes.append(proc2)

        # Give YOLO time to load model
        print("[Launcher] Waiting for YOLO model to load...")
        time.sleep(5)

        # Start Process 1
        print("[Launcher] Starting Process 1 (Weight Capture)...")
        proc1 = subprocess.Popen(
            [sys.executable, PROCESS1_SCRIPT],
            cwd=SCRIPT_DIR
        )
        processes.append(proc1)

        print()
        print("[Launcher] Both processes started!")
        print("[Launcher] Monitoring process status...")
        print()

        # Monitor processes
        while True:
            # Check if either process died
            if proc1.poll() is not None:
                print(f"[Launcher] Process 1 exited with code {proc1.returncode}")
                break

            if proc2.poll() is not None:
                print(f"[Launcher] Process 2 exited with code {proc2.returncode}")
                break

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[Launcher] Ctrl+C received")

    finally:
        print("[Launcher] Terminating processes...")
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)

        print("[Launcher] Done")


if __name__ == '__main__':
    main()
