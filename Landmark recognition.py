"""
Phase 1 — MediaPipe landmark viewer (Tasks API)
NGT Sign Language Demonstrator project
Compatible with mediapipe 0.10.30+

Requires model files in the same folder as this script:
  hand_landmarker.task
  pose_landmarker_lite.task

Download them by running in your terminal:
  curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
  curl -o pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

Controls:
  Q  — quit
  H  — toggle hand landmarks on/off
  P  — toggle pose landmarks on/off
  S  — save a snapshot
"""

import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Connection definitions for drawing ────────────────────────────────────────
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

# ── Display toggles ────────────────────────────────────────────────────────────
show_hands = True
show_pose  = True


def draw_landmarks_on_frame(frame, hand_result, pose_result):
    h, w = frame.shape[:2]

    # Draw pose
    if show_pose and pose_result and pose_result.pose_landmarks:
        for lms in pose_result.pose_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            for a, b in POSE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], (180, 180, 255), 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, (200, 200, 255), -1)

    # Draw hands
    if show_hands and hand_result and hand_result.hand_landmarks:
        for idx, lms in enumerate(hand_result.hand_landmarks):
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            handedness = hand_result.handedness[idx][0].display_name
            color = (100, 220, 100) if handedness == "Right" else (100, 100, 255)
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], color, 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, color, -1)


def extract_keypoints(hand_result, pose_result):
    """Returns a flat 225-value array: 33 pose + 21 left hand + 21 right hand, each (x,y,z)."""
    # Pose
    if pose_result and pose_result.pose_landmarks:
        lms = pose_result.pose_landmarks[0]
        pose = np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten()
    else:
        pose = np.zeros(33 * 3)

    # Separate left/right hands
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


def draw_hud(frame, hand_result, pose_result, fps):
    h, w = frame.shape[:2]

    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    hand_count = len(hand_result.hand_landmarks) if hand_result and hand_result.hand_landmarks else 0
    pose_detected = pose_result and pose_result.pose_landmarks

    indicators = [
        ("Pose",  pose_detected,   show_pose),
        ("Hands", hand_count > 0,  show_hands),
    ]
    for i, (label, detected, visible) in enumerate(indicators):
        color = (80, 220, 80) if detected else (80, 80, 200)
        if not visible:
            color = (100, 100, 100)
        cv2.putText(frame, f"{label}: {'ON' if detected else '-'}", (10, 65 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    cv2.putText(frame, "Q: quit  H: hands  P: pose  S: snapshot",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


def main():
    global show_hands, show_pose

    # ── Load models ───────────────────────────────────────────────────────────
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
    pose_detector = mp_vision.PoseLandmarker.create_from_options(pose_options)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    prev_time = time.time()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run detectors
        hand_result = hand_detector.detect(mp_image)
        pose_result = pose_detector.detect(mp_image)

        # Draw
        draw_landmarks_on_frame(frame, hand_result, pose_result)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        draw_hud(frame, hand_result, pose_result, fps)

        # Extract keypoints (ready for Phase 2)
        keypoints = extract_keypoints(hand_result, pose_result)
        # print(f"Keypoints shape: {keypoints.shape}")  # uncomment to debug

        cv2.imshow("NGT Landmark Viewer — Phase 1", frame)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            show_hands = not show_hands
        elif key == ord('p'):
            show_pose = not show_pose
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