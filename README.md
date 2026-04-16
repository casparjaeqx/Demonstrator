# NGT Gebarentaal Herkenner

Een real-time herkenningssysteem voor Nederlandse Gebarentaal (NGT) gebouwd met MediaPipe en een custom LSTM-neuraal netwerk. Ontwikkeld als een HBO Innovate demonstratorproject.

---

## Hoe het werkt

Het systeem gebruikt een pipeline van vijf fasen:

1. **Landmark-detectie** — MediaPipe extraheert 225 keypoints per frame uit de webcam (33 lichaamspunten + 21 linkerhand + 21 rechterhand, elk met x-, y- en z-coördinaten)  
2. **Data-extractie** — keypoint-sequenties worden uit SignBank-video’s gehaald en opgeslagen als NumPy-arrays  
3. **Augmentatie** — elke sequentie wordt kunstmatig uitgebreid naar ~11 variaties om beperkte trainingsdata te compenseren  
4. **Training** — een LSTM-netwerk met twee lagen leert om sequenties van 30 frames te classificeren naar gebaren  
5. **Live inferentie** — het getrainde model draait op een live webcam-feed en herkent gebaren in real time  

---

## Vereisten

### Python-versie
**Python 3.11** is vereist. MediaPipe ondersteunt geen Python 3.12 of hoger.  
Download: https://www.python.org/downloads/release/python-3119/

### Installatie
```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install mediapipe opencv-python tensorflow scikit-learn requests beautifulsoup4
```

### Modelbestanden
- `hand_landmarker.task`
- `pose_landmarker_lite.task`

---

## Projectstructuur

```
Demonstrator/
├── Landmark recognition.py
├── phase2_extract.py
├── phase3_train.py
├── phase4_inference.py
├── augment.py
├── download_signbank.py
├── record_self.py
├── hand_landmarker.task
├── pose_landmarker_lite.task
├── signbank_cookies.txt
├── ngt.ecv
├── data/
├── dataset/
├── dataset_augmented/
└── model/
    ├── ngt_model.h5
    └── labels.txt
```

---

## Gebruik

### Stap 1 — Download sign videos (Optional)
```bash
Singbank video dowloader.py
```

### Stap 2 — Extract keypoints
```bash
Data extract video.py
```

### Stap 3 — Augment dataset
```bash
Data Argumentation.py
```

### Stap 4 — Train model
```bash
Model train.py
```

### Stap 5 — Live inference
```bash
Live interface.py
```

---

## Nieuwe gebaren toevoegen

1. Download nieuwe data  
2. Extract keypoints  
3. (Optioneel) neem jezelf op  
4. Train opnieuw  

---

## Nauwkeurigheid verbeteren

Gebruik `record_self.py` om eigen data toe te voegen en train opnieuw.

---

## Besturing

| Toets | Actie |
|------|-------|
| Spatie | Voorspellen |
| C | Reset |
| S | Screenshot |
| Q | Quit |

---

## Databronnen

Gebarenvideo’s zijn afkomstig uit de NGT-dataset in Global Signbank:  
https://signbank.cls.ru.nl/datasets/NGT

---

## Tech stack

- MediaPipe  
- OpenCV  
- TensorFlow / Keras  
- NumPy  
- scikit-learn  
- BeautifulSoup  
