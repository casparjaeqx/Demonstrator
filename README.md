# 🤟 NGT Real-Time Gebarentaalherkenning

Een real-time herkenningssysteem voor **Nederlandse Gebarentaal (NGT)** via een gewone webcam. Het systeem combineert **MediaPipe** voor handlandmark-detectie met een **LSTM-neuraal netwerk** voor temporele gebarenclassificatie.

> HBO-ICT innovator/demonstratorproject — gebouwd in Python 3.11

---

## 📋 Inhoudsopgave

- [Overzicht](#overzicht)
- [Pipeline](#pipeline)
- [Projectstructuur](#projectstructuur)
- [Installatie](#installatie)
- [Gebruik](#gebruik)
- [Data verzamelen](#data-verzamelen)
- [Model trainen](#model-trainen)
- [Live inferentie](#live-inferentie)
- [Technische details](#technische-details)
- [Afhankelijkheden](#afhankelijkheden)

---

## Overzicht

Dit project herkent dynamische NGT-gebaren in real-time via een webcam. Het pipeline verwerkt videoframes naar skelet-landmarks, slaat keypoint-sequenties op, traint een LSTM-model en voert live classificatie uit.

De trainingsdata is afkomstig uit de **SignBank NGT-database** (signbank.cls.ru.nl), waarbij per gebaar video-opnames vanuit drie camerahoeken worden gebruikt.

---

## Pipeline

```
Webcam / Video
      │
      ▼
MediaPipe Landmarks          ← hand (21 punten × 2) + pose (33 punten)
      │
      ▼
Keypoint-extractie           ← (30 frames × 225 features) per sequentie
      │
      ▼
Data-augmentatie             ← ×11 per originele sequentie
      │
      ▼
LSTM-training                ← TensorFlow/Keras
      │
      ▼
Live inferentie              ← real-time classificatie via webcam
```

---

## Projectstructuur

```
ngt-herkenning/
├── data/                        # Trainingsdata (per gebaar een submap)
│   ├── hallo/
│   ├── bedankt/
│   └── ...
├── models/                      # Getrainde LSTM-modellen (.h5)
├── scripts/
│   ├── landmark_viewer.py       # Fase 1: live landmark-visualisatie
│   ├── extract_keypoints.py     # Fase 2: keypoints uit video's extraheren
│   ├── train_lstm.py            # Fase 3: LSTM-model trainen
│   ├── live_inference.py        # Fase 4: live herkenning via webcam
│   ├── signbank_downloader.py   # SignBank video-downloader (cookie-auth)
│   └── augment_data.py          # Data-augmentatie
├── tasks/                       # MediaPipe Task-modelbestanden
│   ├── hand_landmarker.task
│   └── pose_landmarker_lite.task
├── requirements.txt
└── README.md
```

---

## Installatie

> ⚠️ **Python 3.11 is vereist.** MediaPipe is niet compatibel met Python 3.12+.

### 1. Virtuele omgeving aanmaken

```bash
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Afhankelijkheden installeren

```bash
pip install -r requirements.txt
```

### 3. MediaPipe Task-modellen downloaden

Download de benodigde modelbestanden en plaats ze in de `tasks/` map:

- [`hand_landmarker.task`](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
- [`pose_landmarker_lite.task`](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task)

---

## Gebruik

### Fase 1 — Landmark-visualisatie

Controleer of MediaPipe correct werkt door landmarks live op het webcambeeld te tekenen:

```bash
python scripts/landmark_viewer.py
```

### Fase 2 — Data extractie uit video's

Verwerk video's in `data/<gebaar>/` naar keypoint-sequenties:

```bash
python scripts/extract_keypoints.py
```

Per gebaar worden `.npy`-bestanden opgeslagen als arrays van vorm `(30, 225)`.

### Fase 3 — Model trainen

Train het LSTM-model op de geëxtraheerde keypoints:

```bash
python scripts/train_lstm.py
```

Het getrainde model wordt opgeslagen in `models/`.

### Fase 4 — Live inferentie

Start real-time gebarenherkenning via de webcam:

```bash
python scripts/live_inference.py
```

---

## Data verzamelen

### SignBank downloader

Video's worden automatisch gedownload vanuit de NGT SignBank-database. Omdat SignBank inloggen vereist, gebruik je geëxporteerde browsercookies (Netscape-formaat).

1. Log in op [signbank.cls.ru.nl](https://signbank.cls.ru.nl) in je browser
2. Exporteer je cookies als `cookies.txt` (bijv. met de extensie *Get cookies.txt LOCALLY*)
3. Plaats `cookies.txt` in de projectroot
4. Voer de downloader uit:

```bash
python scripts/signbank_downloader.py
```

Per gebaar worden opnames vanuit drie camerahoeken (midden, links, rechts) gedownload.

---

## Model trainen

Het LSTM-model verwerkt sequenties van **30 frames**, elk bestaande uit:

| Bron | Punten | Features |
|---|---|---|
| Linkerhand (MediaPipe) | 21 | 63 (x, y, z) |
| Rechterhand (MediaPipe) | 21 | 63 (x, y, z) |
| Pose (MediaPipe) | 33 | 99 (x, y, z) |
| **Totaal** | | **225 per frame** |

### Data-augmentatie

Per originele sequentie worden automatisch **11 varianten** gegenereerd via:

- Gaussische ruis
- Schaling
- Snelheidsvariatie
- Tijdsverschuiving
- Rotatie
- Spiegeling
- Willekeurige combinaties

---

## Technische details

| Onderdeel | Keuze | Reden |
|---|---|---|
| Landmark-detectie | MediaPipe Tasks API v0.10.33 | Efficiënte twee-staps pipeline, 21 handpunten |
| Tijdreeksclassificatie | LSTM (Keras) | Geschikt voor variabele gebaardynamiek |
| Invoerformaat | `(30, 225)` numpy-arrays | 30 frames, 225 features per frame |
| Python-versie | 3.11 | MediaPipe-compatibiliteit |

---

## Afhankelijkheden

```
mediapipe==0.10.33
opencv-python
tensorflow
scikit-learn
numpy
requests
browser-cookie3
```

---

## Licentie

Dit project is ontwikkeld als onderdeel van een HBO-ICT studieopdracht. De trainingsdata is afkomstig uit de NGT SignBank-database van de Radboud Universiteit Nijmegen en Universiteit Amsterdam.