# NGT Sign Language Demonstrator — Phase 1

A real-time landmark viewer for NGT (Dutch Sign Language) recognition, built with MediaPipe and OpenCV.

---

## Requirements

### Python version
**Python 3.11** is required. MediaPipe does not support Python 3.12 or higher.
Download: https://www.python.org/downloads/release/python-3119/

### Python packages
Install inside a virtual environment:
```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install mediapipe opencv-python
```

### Model files
The required model files are included in the repository — no download needed:
- `hand_landmarker.task`
- `pose_landmarker_lite.task`

Make sure they stay in the **same folder as the script**.

---

## Running the script

Make sure the venv is active (you should see `(venv)` in your terminal), then:
```bash
python "Landmark recognition.py"
```

---

## Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `H` | Toggle hand landmarks on/off |
| `P` | Toggle pose landmarks on/off |
| `S` | Save a snapshot of the current frame |

---

## Changing the camera

On line 141 of the script, change the camera index if needed:
```python
cap = cv2.VideoCapture(0)  # 0 = default, try 1 or 2 for other cameras
```

To find which index your camera uses, run:
```python
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
```

---

## Project structure

```
Demonstrator/
├── Landmark recognition.py     # Main script
├── hand_landmarker.task        # Hand detection model
├── pose_landmarker_lite.task   # Pose detection model
├── venv/                       # Python 3.11 virtual environment
└── README.md                   # This file
```

---

## Troubleshooting

**`No module named mediapipe`** — make sure the venv is activated before running the script.

**`Could not open webcam`** — try changing the camera index from `0` to `1`.

**Slow performance** — the script uses `model_complexity=1` by default. Change it to `0` for faster but less accurate detection.