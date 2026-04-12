# ⚽ Football Tracking & Highlight Generation

End-to-end pipeline for detecting the ball, tracking players, identifying events, and generating highlight videos from football matches.

---

## 🚀 Features

* 🎯 Ball detection using YOLO
* 🧠 Player tracking and interpolation
* 🟦 Team assignment based on proximity
* ⚡ Event detection (kickoff, passes, etc.)
* 🎬 Automatic highlight generation
* 📦 Optional highlight concatenation

---

## 📂 Project Structure

```
football_tracking/        # Core pipeline logic
notebooks/                # Colab notebooks
scripts/                  # Local execution scripts

input_videos/             # (empty) place your videos here
output_videos/            # Generated outputs
models/                   # Trained models
datasets/                 # Training datasets
rois/                     # Region of Interest configs
```

---

## ⚙️ Setup

```bash
git clone <your-repo-url>
cd Football-Tracking

pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Add videos

Place your `.mp4` files into:

```
input_videos/
```

---

## 🎥 Demo — Ball Tracking & Player Association

This demo shows the pipeline in action:

* ⚽ Ball detection with interpolation when missing
* 🧍 Player tracking across frames
* 🔗 Association between ball and closest player
* 🎯 Foundation for event detection and highlight generation

👉 **Watch the full demo video:**
https://youtu.be/78WEZ-UguGs

---

### 2. Run pipeline

```bash
python scripts/run_local.py
```

Outputs will appear in:

```
output_videos/
```

---

### 3. (Optional) Concatenate highlights

Use the notebook:

```
notebooks/utils/03_concatenate_highlights.ipynb
```

---

## 🧠 Model Training (optional)

Training is done separately using:

```
notebooks/training/01_ball_model_training.ipynb
```

You only need to retrain if improving detection performance.

---

## 📊 Pipeline Overview

1. Ball detection
2. Player tracking
3. Ball interpolation
4. Team possession inference
5. Event detection
6. Highlight generation

---

## ⚠️ Notes

* Large files (videos, datasets, models) are **not tracked in Git**
* Use local folders or cloud storage for assets
* Ensure ROI files exist before running the pipeline

---

## 🛠️ Tech Stack

* Python
* OpenCV
* YOLO (Ultralytics)
* NumPy / Pandas
* FFmpeg

---

## 📌 Future Improvements

* Real-time processing
* Improved team classification
* Advanced event detection (goals, fouls)
* Web interface / dashboard

---


## 👤 Author

Marco De Stavola
