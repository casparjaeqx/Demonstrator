"""
Phase 4 — Live Inference
NGT Sign Language Demonstrator project

Connects the webcam to the trained LSTM model and displays
the predicted sign in real time.

Requires:
  model/ngt_model.h5        — trained model (output of Phase 3)
  model/labels.txt          — sign labels (output of Phase 3)
  hand_landmarker.task      — MediaPipe hand model
  pose_landmarker_lite.task — MediaPipe pose model

Usage:
  python phase4_inference.py

Controls:
  SPACE  — start capturing a 30-frame window for prediction
  C      — clear current result
  Q      — quit
  S      — save a snapshot
"""

import cv2
import numpy as np
import time
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tensorflow.keras.models import load_model
from collections import deque

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join("model", "ngt_model.h5")
LABELS_PATH    = os.path.join("model", "labels.txt")
SEQUENCE_LEN   = 30
NUM_FEATURES   = 225
CONFIDENCE_MIN = 0.7    # minimum confidence to show a result
DISPLAY_SIZE   = (1280, 720)

# ── Connection definitions ─────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),
]


# ── Keypoint extraction ────────────────────────────────────────────────────────
def extract_keypoints(hand_result, pose_result):
    pose = (
        np.array([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]]).flatten()
        if pose_result and pose_result.pose_landmarks else np.zeros(33 * 3)
    )
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    if hand_result and hand_result.hand_landmarks:
        for idx, lms in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[idx][0].display_name
            arr = np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten()
            if handedness == "Left":
                lh = arr
            else:
                rh = arr
    return np.concatenate([pose, lh, rh])


# ── Landmark drawing ───────────────────────────────────────────────────────────
def draw_landmarks(frame, hand_result, pose_result):
    h, w = frame.shape[:2]
    if pose_result and pose_result.pose_landmarks:
        for lms in pose_result.pose_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            for a, b in POSE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], (180, 180, 255), 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, (200, 200, 255), -1)
    if hand_result and hand_result.hand_landmarks:
        for idx, lms in enumerate(hand_result.hand_landmarks):
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            handedness = hand_result.handedness[idx][0].display_name
            color = (100, 220, 100) if handedness == "Right" else (100, 100, 255)
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], color, 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, color, -1)


# ── Draw overlay ───────────────────────────────────────────────────────────────
def draw_overlay(frame, label, confidence, capturing, capture_buffer):
    h, w = frame.shape[:2]

    if capturing:
        # Progress bar at top
        progress = len(capture_buffer) / SEQUENCE_LEN
        bar_w    = int(w * progress)
        cv2.rectangle(frame, (0, 0), (w, 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (0, 0), (bar_w, 10), (80, 200, 80), -1)

        # Recording indicator
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 180), 3)
        frames_left = SEQUENCE_LEN - len(capture_buffer)
        cv2.putText(frame, f"Capturing... {frames_left} frames left",
                    (w // 2 - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

    elif label:
        # Result box at bottom
        box_h = 90
        cv2.rectangle(frame, (0, h - box_h), (w, h), (20, 20, 20), -1)
        cv2.rectangle(frame, (0, h - box_h), (w, h), (60, 60, 60), 1)

        # Sign label centred
        font_scale = 1.8
        thickness  = 3
        text_size  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x     = (w - text_size[0]) // 2
        cv2.putText(frame, label, (text_x, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Confidence bar
        conf_bar_w = int((w - 40) * confidence)
        cv2.rectangle(frame, (20, h - 18), (w - 20, h - 8), (60, 60, 60), -1)
        color = (80, 220, 80) if confidence > 0.85 else (80, 180, 220)
        cv2.rectangle(frame, (20, h - 18), (20 + conf_bar_w, h - 8), color, -1)
        cv2.putText(frame, f"{confidence * 100:.0f}%", (w - 55, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    else:
        # Idle prompt
        cv2.rectangle(frame, (0, h - 50), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, "Press SPACE to sign",
                    (w // 2 - 130, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 200, 120), 2)

def draw_logo(frame, logo):
    if logo is None:
        return
    h, w = frame.shape[:2]
    lh, lw = logo.shape[:2]
    # Position: top right with 10px padding
    x = w - lw - 10
    y = 10
    # Handle transparency if PNG, otherwise just overlay
    frame[y:y+lh, x:x+lw] = logo

def draw_hud(frame, fps, num_hands, pose_detected):
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    hand_color = (80, 220, 80) if num_hands > 0 else (80, 80, 200)
    pose_color = (80, 220, 80) if pose_detected else (80, 80, 200)
    cv2.putText(frame, f"Hands: {num_hands}", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)
    cv2.putText(frame, f"Pose: {'yes' if pose_detected else 'no'}", (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, pose_color, 1)
    h = frame.shape[0]
    cv2.putText(frame, "SPACE: sign  C: clear  S: snapshot  Q: quit",
                (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if not os.path.exists(MODEL_PATH):
        print(f"Error: model not found at '{MODEL_PATH}'. Run phase3_train.py first.")
        return
    if not os.path.exists(LABELS_PATH):
        print(f"Error: labels not found at '{LABELS_PATH}'. Run phase3_train.py first.")
        return

    print("Loading model...")
    model = load_model(MODEL_PATH)

    # Load logo
    logo_path = "ZUYD_LOGO.jpg"
    logo = None
    if os.path.exists(logo_path):
        raw   = cv2.imread(logo_path)
        scale = 80 / raw.shape[0]   # resize to 80px tall
        logo  = cv2.resize(raw, (int(raw.shape[1] * scale), 80))

    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    print(f"Loaded model with {len(labels)} signs: {', '.join(labels)}\n")

    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    hand_detector = mp_vision.HandLandmarker.create_from_options(hand_options)
    pose_detector  = mp_vision.PoseLandmarker.create_from_options(pose_options)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    capturing      = False
    capture_buffer = []
    current_label      = None
    current_confidence = 0.0
    prev_time          = time.time()

    print("Ready — press SPACE to start a capture window, Q to quit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        hand_result = hand_detector.detect(mp_image)
        pose_result  = pose_detector.detect(mp_image)

        keypoints = extract_keypoints(hand_result, pose_result)

        # Collect frames only when capturing
        if capturing:
            capture_buffer.append(keypoints)

            if len(capture_buffer) >= SEQUENCE_LEN:
                # Run prediction on the captured window
                sequence   = np.array(capture_buffer[:SEQUENCE_LEN])
                input_data = np.expand_dims(sequence, axis=0)
                predictions = model.predict(input_data, verbose=0)[0]
                top_idx     = np.argmax(predictions)
                confidence  = float(predictions[top_idx])

                if confidence >= CONFIDENCE_MIN:
                    current_label      = labels[top_idx]
                    current_confidence = confidence
                    print(f"Predicted: {current_label} ({confidence * 100:.1f}%)")
                else:
                    current_label      = None
                    current_confidence = 0.0
                    print(f"Low confidence ({confidence * 100:.1f}%) — no prediction shown")

                capture_buffer = []
                capturing      = False

        # Draw landmarks on original frame
        draw_landmarks(frame, hand_result, pose_result)

        # Resize then draw UI
        frame = cv2.resize(frame, DISPLAY_SIZE)

        num_hands     = len(hand_result.hand_landmarks) if hand_result and hand_result.hand_landmarks else 0
        pose_detected = bool(pose_result and pose_result.pose_landmarks)

        now       = time.time()
        fps       = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        draw_hud(frame, fps, num_hands, pose_detected)
        draw_logo(frame, logo)
        draw_overlay(frame, current_label, current_confidence, capturing, capture_buffer)

        cv2.imshow("NGT Sign Recognition — Phase 4", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if not capturing:
                capture_buffer = []
                capturing      = True
                current_label  = None
                print("Capturing...")
        elif key == ord('c'):
            capturing      = False
            capture_buffer = []
            current_label      = None
            current_confidence = 0.0
        elif key == ord('s'):
            filename = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"Snapshot saved: {filename}")

    hand_detector.close()
    pose_detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()