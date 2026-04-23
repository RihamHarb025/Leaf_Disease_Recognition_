# LeafGuard — Leaf Disease Recognition App

**Team:** Riham Harb (20230088) · Abdulrhman Misto (20240359) · Adel Soufian (20230174)  
**Course:** COSC498 — Image Processing · Rafik Hariri University

---

## Project Structure

```
leafguard/
├── app.py              ← Streamlit UI (all pages in one file)
├── helpers.py          ← Plotting + I/O utilities
├── train.py            ← CLI training script
├── test_pipeline.py    ← 18 unit tests
├── requirements.txt
│
├── pipeline/           ← Core image processing pipeline
│   ├── preprocessing.py    quality checks + resize/denoise/CLAHE
│   ├── segmentation.py     leaf mask + stress mask
│   ├── features.py         28-feature vector extraction
│   ├── classifier.py       k-NN + LOO cross-validation
│   ├── decision_engine.py  sensor fusion + actuator logic
│   └── dataset.py          dataset loader + train_from_folder()
│
├── models/             ← Saved model (generated after training)
│   ├── knn_model.pkl
│   └── knn_meta.json
│
└── data/logs/          ← Session log CSV
```

---

## Run the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app works immediately without training — it uses a rule-based fallback until you train a model.

---

## Train a Model

**Option A — GUI (recommended):**  
Open the app → **🎓 Train** tab → upload a dataset ZIP → click Train.

**Option B — Command line:**
```bash
python train.py --dataset dataset_imgs/
python train.py --dataset dataset_imgs/ --k 7 --max-per-class 150
```

---

## Dataset Format

Place your dataset in a folder with one sub-folder per class:

```
dataset_imgs/
    Tomato___healthy/
    Tomato___Early_blight/
    Tomato___Leaf_Miner/
    Pepper__bell___Bacterial_spot/
    ...
```

Supported: **PlantVillage** and **plant_leaves** datasets.  
Folder names are automatically mapped to 5 canonical classes via keyword matching.

---

## Pipeline

```
Image → Quality Check → Preprocess → Segment Leaf → Segment Stress
      → Extract 28 Features → k-NN Classify → Fuse with Sensors → Decision
```

| Stage | Method |
|---|---|
| Quality check | Brightness / contrast / Laplacian blur / HF energy |
| Preprocessing | Resize 512×512, bilateral denoise, CLAHE contrast |
| Leaf segmentation | HSV green thresholding + Otsu fallback |
| Stress segmentation | Multi-cue HSV + LAB thresholding |
| Features | 10 morphology + 10 colour + 6 texture (LBP, GLCM) + 2 edge |
| Classification | k-NN (k=5, distance-weighted, Z-score normalised) |
| Evaluation | Leave-One-Out cross-validation |
| Sensor fusion | pH / EC / water level / temperature / humidity |

---

## Run Tests

```bash
python -m pytest test_pipeline.py -v
```
