"""
Phase 2 — Keypoint extraction from video files
NGT Sign Language Demonstrator project

Folder structure expected:
  data/
  ├── SIGN_ONE/
  │   ├── video1.mp4
  │   └── video2.mp4
  ├── SIGN_TWO/
  │   └── video1.mp4
  └── ...

Output:
  dataset/
  ├── SIGN_ONE/
  │   ├── 0.npy
  │   └── 1.npy
  ├── SIGN_TWO/
  │   └── 0.npy
  └── ...

Each .npy file is a (30, 225) array — 30 frames × 225 keypoints.

Usage:
  python phase2_extract.py
"""

import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"       # folder containing one subfolder per sign
OUTPUT_DIR  = "dataset"    # where keypoint .npy files will be saved
SEQUENCE_LEN = 30          # number of frames per sequence

# ── MediaPipe setup ────────────────────────────────────────────────────────────
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


# ── Keypoint extraction (same as Phase 1) ─────────────────────────────────────
def extract_keypoints(hand_result, pose_result):
    """Returns a flat 225-value array: 33 pose + 21 left hand + 21 right hand (x,y,z each)."""
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


# ── Process a single video into a (SEQUENCE_LEN, 225) array ───────────────────
def process_video(video_path):
    """
    Extracts SEQUENCE_LEN evenly spaced frames from a video,
    runs MediaPipe on each, and returns a (SEQUENCE_LEN, 225) numpy array.
    Returns None if the video can't be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [!] Could not open: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        print(f"  [!] Empty video: {video_path}")
        cap.release()
        return None

    # Pick SEQUENCE_LEN evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, SEQUENCE_LEN, dtype=int)

    sequence = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # If frame read fails, use zeros
            sequence.append(np.zeros(225))
            continue

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        hand_result = hand_detector.detect(mp_image)
        pose_result  = pose_detector.detect(mp_image)
        sequence.append(extract_keypoints(hand_result, pose_result))

    cap.release()
    return np.array(sequence)  # shape: (SEQUENCE_LEN, 225)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Find all sign subfolders in data/
    if not os.path.exists(DATA_DIR):
        print(f"Error: '{DATA_DIR}' folder not found. Create it and add sign subfolders.")
        return

    signs = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    if not signs:
        print(f"No subfolders found in '{DATA_DIR}'. Add a folder per sign.")
        return

    print(f"Found {len(signs)} sign(s): {', '.join(signs)}\n")

    total_saved = 0
    total_failed = 0

    for sign in signs:
        sign_dir   = os.path.join(DATA_DIR, sign)
        output_dir = os.path.join(OUTPUT_DIR, sign)
        os.makedirs(output_dir, exist_ok=True)

        # Find all video files for this sign
        videos = sorted([
            f for f in os.listdir(sign_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ])

        if not videos:
            print(f"[{sign}] No video files found, skipping.")
            continue

        print(f"[{sign}] Processing {len(videos)} video(s)...")

        for i, video_file in enumerate(videos):
            video_path = os.path.join(sign_dir, video_file)
            print(f"  ({i+1}/{len(videos)}) {video_file}", end=" ... ")

            sequence = process_video(video_path)

            if sequence is not None:
                out_path = os.path.join(output_dir, f"{i}.npy")
                np.save(out_path, sequence)
                print(f"saved → {out_path}")
                total_saved += 1
            else:
                print("failed, skipped")
                total_failed += 1

    print(f"\nDone. {total_saved} sequences saved, {total_failed} failed.")
    print(f"Dataset ready in '{OUTPUT_DIR}/'")

    # Print a summary of what was collected
    print("\nSummary:")
    for sign in signs:
        sign_output = os.path.join(OUTPUT_DIR, sign)
        if os.path.exists(sign_output):
            count = len([f for f in os.listdir(sign_output) if f.endswith(".npy")])
            print(f"  {sign}: {count} sequence(s)")


if __name__ == "__main__":
    main()