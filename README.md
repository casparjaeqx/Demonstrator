# NGT Sign Language Recogniser

A real-time Dutch Sign Language (NGT — Nederlandse Gebarentaal) recognition system built with MediaPipe and a custom LSTM neural network. Developed as an HBO Inovate demonstrator project.

---

## How it works

The system uses a five-phase pipeline:

1. **Landmark detection** — MediaPipe extracts 225 keypoints per frame from the webcam feed (33 body pose points + 21 left hand + 21 right hand, each with x, y, z coordinates)
2. **Data extraction** — keypoint sequences are extracted from SignBank videos and saved as numpy arrays
3. **Augmentation** — each sequence is artificially expanded into ~11 variations to compensate for limited training data
4. **Training** — a two-layer LSTM network learns to classify 30-frame sequences into sign labels
5. **Live inference** — the trained model runs on a live webcam feed and predicts signs in real time

---

## Requirements

### Python version
**Python 3.11** is required. MediaPipe does not support Python 3.12 or higher.
Download: https://www.python.org/downloads/release/python-3119/

### Setup
```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install mediapipe opencv-python tensorflow scikit-learn requests beautifulsoup4
```

### Model files
The following MediaPipe model files are included in the repository:
- `hand_landmarker.task`
- `pose_landmarker_lite.task`

---

## Project structure

```
Demonstrator/
├── Landmark recognition.py     # Phase 1 — live landmark viewer
├── phase2_extract.py           # Phase 2 — extract keypoints from videos
├── phase3_train.py             # Phase 3 — train the LSTM model
├── phase4_inference.py         # Phase 4 — live sign recognition
├── augment.py                  # Data augmentation
├── download_signbank.py        # Download videos from SignBank
├── record_self.py              # Record your own signing for training
├── hand_landmarker.task        # MediaPipe hand model
├── pose_landmarker_lite.task   # MediaPipe pose model
├── signbank_cookies.txt        # SignBank session cookies (not committed)
├── ngt.ecv                     # NGT sign database
├── data/                       # Downloaded videos (one subfolder per sign)
├── dataset/                    # Extracted keypoint sequences
├── dataset_augmented/          # Augmented sequences
└── model/                      # Trained model and labels
    ├── ngt_model.h5
    └── labels.txt
```

---

## Usage

### Step 1 — Download sign videos from SignBank
```bash
python download_signbank.py
```
Requires a `signbank_cookies.txt` file exported from your browser while logged into SignBank. Use a browser extension such as "Get cookies.txt LOCALLY" (Chrome) or "cookies.txt" (Firefox).

### Step 2 — Extract keypoints from videos
```bash
python phase2_extract.py
```
Processes all videos in `data/` and saves keypoint sequences to `dataset/`.

### Step 3 — Augment the dataset
```bash
python augment.py
```
Expands the dataset from `dataset/` to `dataset_augmented/` using 7 augmentation techniques.

### Step 4 — Train the model
```bash
python phase3_train.py
```
Trains an LSTM on the augmented dataset and saves the model to `model/`.

### Step 5 — Run live inference
```bash
python phase4_inference.py
```
Opens the webcam feed. Press **Space** to capture a 30-frame window and get a prediction.

---

## Adding new signs

1. Run `download_signbank.py` and search for the new sign
2. Run `phase2_extract.py` to extract keypoints
3. Optionally record yourself signing with `record_self.py`
4. Run `augment.py` then `phase3_train.py` to retrain

---

## Improving accuracy

If the model predicts the wrong sign, the most effective fix is to record yourself performing each sign using `record_self.py` and retrain. The model is initially trained on SignBank videos which may look different from your own signing.

```bash
python record_self.py
```

Use number keys to select a sign, press **R** to record a 30-frame sequence, repeat 10–15 times per sign, then retrain.

---

## Controls

### Live inference (`phase4_inference.py`)
| Key | Action |
|-----|--------|
| `Space` | Capture a 30-frame window and predict |
| `C` | Clear current result |
| `S` | Save a snapshot |
| `Q` | Quit |

### Landmark viewer (`Landmark recognition.py`)
| Key | Action |
|-----|--------|
| `H` | Toggle hand landmarks |
| `P` | Toggle pose landmarks |
| `S` | Save a snapshot |
| `Q` | Quit |

### Self-recording (`record_self.py`)
| Key | Action |
|-----|--------|
| `1–9` | Select sign |
| `R` | Start/cancel recording |
| `Q` | Quit |

---

## Data sources

Sign videos are sourced from the [NGT dataset in Global Signbank](https://signbank.cls.ru.nl/datasets/NGT), a lexical database developed by Radboud University Nijmegen and the University of Amsterdam. Access requires registration.

---

## Tech stack

| Component | Library |
|-----------|---------|
| Landmark detection | MediaPipe 0.10.33 |
| Video / webcam | OpenCV |
| Neural network | TensorFlow / Keras (LSTM) |
| Data processing | NumPy, scikit-learn |
| SignBank scraping | requests, BeautifulSoup |
