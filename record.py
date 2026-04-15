"""
Self-recording data collection tool
NGT Sign Language Demonstrator project

Records your own signing directly into the dataset folder
so the model can learn what your signing looks like.

Controls:
  R        — start/stop recording a sequence
  1-9      — select sign by number from the list
  Q        — quit
"""

import cv2
import numpy as np
import os
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset"
SEQUENCE_LEN = 30
NUM_FEATURES = 225
DISPLAY_SIZE = (1280, 720)

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


def get_next_index(sign_dir):
    existing = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]
    return len(existing)


def draw_ui(frame, signs, selected_idx, recording, record_buffer, recordings_per_sign):
    h, w = frame.shape[:2]

    # Left panel background
    cv2.rectangle(frame, (0, 0), (220, h), (20, 20, 20), -1)
    cv2.putText(frame, "Signs:", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    for i, sign in enumerate(signs):
        y = 60 + i * 32
        count = recordings_per_sign.get(sign, 0)
        is_selected = i == selected_idx
        bg_color   = (60, 120, 60) if is_selected else (30, 30, 30)
        text_color = (255, 255, 255) if is_selected else (160, 160, 160)
        cv2.rectangle(frame, (5, y - 18), (215, y + 8), bg_color, -1)
        cv2.putText(frame, f"[{i+1}] {sign} ({count})", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    if recording:
        progress = len(record_buffer) / SEQUENCE_LEN
        bar_w    = int((w - 240) * progress)
        bar_y    = h - 60
        cv2.rectangle(frame, (220, 0), (w, h), (0, 0, 180), 3)
        cv2.rectangle(frame, (230, bar_y), (w - 10, bar_y + 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (230, bar_y), (230 + bar_w, bar_y + 20), (0, 0, 200), -1)
        frames_left = SEQUENCE_LEN - len(record_buffer)
        cv2.putText(frame, f"RECORDING — {frames_left} frames left",
                    (230, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)
    else:
        sign_name = signs[selected_idx] if signs else "—"
        cv2.putText(frame, f"Selected: {sign_name}",
                    (230, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "Press R to record",
                    (230, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 200, 120), 1)

    cv2.putText(frame, "R: record  1-9: select sign  Q: quit",
                (230, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)


def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: '{DATASET_DIR}' folder not found.")
        print("Run phase2_extract.py first to create the dataset folder structure.")
        return

    signs = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    if not signs:
        print(f"No sign folders found in '{DATASET_DIR}'.")
        return

    print(f"Found {len(signs)} sign(s): {', '.join(signs)}")
    print("Use number keys to select a sign, R to record, Q to quit.\n")

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

    selected_idx  = 0
    recording     = False
    record_buffer = []

    recordings_per_sign = {
        sign: len([f for f in os.listdir(os.path.join(DATASET_DIR, sign)) if f.endswith(".npy")])
        for sign in signs
    }

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

        # Draw landmarks on original resolution frame
        draw_landmarks(frame, hand_result, pose_result)

        # Extract keypoints and collect if recording
        keypoints = extract_keypoints(hand_result, pose_result)

        if recording:
            record_buffer.append(keypoints)

            if len(record_buffer) >= SEQUENCE_LEN:
                sign      = signs[selected_idx]
                sign_dir  = os.path.join(DATASET_DIR, sign)
                idx       = get_next_index(sign_dir)
                save_path = os.path.join(sign_dir, f"{idx}.npy")

                sequence = np.array(record_buffer[:SEQUENCE_LEN])
                np.save(save_path, sequence)

                recordings_per_sign[sign] = recordings_per_sign.get(sign, 0) + 1
                print(f"Saved: {save_path} ({recordings_per_sign[sign]} total for {sign})")

                record_buffer = []
                recording     = False

        # Resize frame then draw UI on top
        frame = cv2.resize(frame, DISPLAY_SIZE)
        draw_ui(frame, signs, selected_idx, recording, record_buffer, recordings_per_sign)

        cv2.imshow("Self-recording — NGT Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            if not recording:
                record_buffer = []
                recording     = True
                print(f"Recording {signs[selected_idx]}...")
            else:
                recording     = False
                record_buffer = []
                print("Recording cancelled.")
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(signs):
                selected_idx  = idx
                recording     = False
                record_buffer = []

    hand_detector.close()
    pose_detector.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\nRecording session complete.")
    print("Run augment.py and then phase3_train.py to retrain with your new data.")


if __name__ == "__main__":
    main()