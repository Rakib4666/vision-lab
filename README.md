# Vision-Lab 🖼️🎥

A comprehensive **Computer Vision portfolio** showcasing end‑to‑end projects in **image classification, object detection (YOLOv8, Faster R‑CNN), segmentation, and real‑time video analysis** using **PyTorch, TensorFlow/Keras, OpenCV, and Ultraly tics**. Built to be both **recruiter‑friendly** and **technically rigorous**.

---

## Table of Contents

* [Highlights](#highlights)
* [Project Index](#project-index)
* [Directory Layout](#directory-layout)
* [Quickstart](#quickstart)

  * [Install dependencies](#install-dependencies)
  * [Run a 30‑second smoke test](#run-a-30-second-smoke-test)
* [Core Recipes](#core-recipes)

  * [Real‑time webcam detection (YOLOv8 + OpenCV)](#real-time-webcam-detection-yolov8--opencv)
  * [Image segmentation demo (YOLOv8‑Seg)](#image-segmentation-demo-yolov8-seg)
  * [Image classification: ResNet‑18 on CIFAR‑10 (PyTorch)](#image-classification-resnet-18-on-cifar-10-pytorch)
  * [Streamlit image detector app](#streamlit-image-detector-app)
* [Datasets](#datasets)
* [Reproducibility](#reproducibility)
* [Performance Notes](#performance-notes)
* [License](#license)
* [Author](#author)

---

## Highlights

* **Breadth + depth**: classification, detection, segmentation, video, and real‑time apps.
* **Production‑oriented**: clean project layout, reusable `src/` modules, and a Streamlit demo.
* **Fast to demo**: one‑line commands to verify everything works.
* **Extensible**: add new datasets, swap models, or plug into FastAPI for serving.

> Topics: `computer-vision`, `deep-learning`, `object-detection`, `image-classification`, `image-segmentation`, `video-analysis`, `opencv`, `pytorch`, `tensorflow`, `yolov8`, `streamlit`, `mlops`.

---

## Project Index

* **Detection (YOLOv8)**: quickstart inference + training on `coco128` sample.
* **Segmentation (YOLOv8‑Seg)**: instance/semantic mask predictions on images.
* **Classification (ResNet‑18)**: transfer learning on CIFAR‑10 with a concise PyTorch script.
* **Real‑time**: webcam detector with OpenCV; Streamlit app for drag‑and‑drop images.

---

## Directory Layout

```
vision-lab/
├─ notebooks/
│  ├─ detection_yolov8_coco128.ipynb
│  ├─ segmentation_yolov8_seg.ipynb
│  ├─ classification_resnet_cifar10.ipynb
│  └─ video_webcam_yolov8.ipynb
├─ src/
│  ├─ detection/
│  │  ├─ train_yolov8.py
│  │  └─ infer_webcam_yolo.py
│  ├─ segmentation/
│  │  └─ train_yolov8_seg.py
│  ├─ classification/
│  │  └─ train_resnet_cifar10.py
│  └─ utils/
│     └─ viz.py
├─ apps/
│  └─ streamlit_yolo_app/
│     └─ app.py
├─ datasets/            # Auto‑downloaded by scripts/notebooks where possible
├─ requirements.txt
├─ LICENSE
└─ README.md
```

> You can create these files/folders as you go. Each recipe below includes fully working code you can drop into the indicated path.

---

## Quickstart

```bash
# (Optional) create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Clone (after you create the repo on GitHub)
git clone https://github.com/iamvisheshsrivastava/vision-lab.git
cd vision-lab
```

### Install dependencies

Create `requirements.txt` with the following content:

```txt
ultralytics>=8.2.0
opencv-python
torch
torchvision
torchaudio
numpy
pandas
matplotlib
albumentations
pillow
tqdm
streamlit
fastapi
uvicorn
```

Then install:

```bash
pip install -r requirements.txt
```

> **GPU users:** install a CUDA‑enabled PyTorch build from [https://pytorch.org](https://pytorch.org) for faster training/inference.

### Run a 30‑second smoke test

Object detection on a sample image with pretrained YOLOv8n:

```bash
yolo predict model=yolov8n.pt source=https://ultralytics.com/images/bus.jpg
```

The annotated image will be saved under `runs/detect/predict*`.

---

## Core Recipes

### Real‑time webcam detection (YOLOv8 + OpenCV)

Create `src/detection/infer_webcam_yolo.py` with:

```python
from ultralytics import YOLO
import cv2

# Load a small, fast model for real-time use
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame)  # inference
    annotated = results[0].plot()  # draw boxes, labels, conf

    cv2.imshow("YOLOv8 Webcam", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Run it:

```bash
python src/detection/infer_webcam_yolo.py
```

Press `q` to quit.

---

### Image segmentation demo (YOLOv8‑Seg)

Predict segmentation masks on a sample image:

```bash
yolo segment predict model=yolov8n-seg.pt source=https://ultralytics.com/images/bus.jpg
```

Train a quick segmentation model on the `coco128-seg` sample:

```bash
yolo segment train model=yolov8n-seg.pt data=coco128-seg.yaml epochs=10 imgsz=640
```

---

### Image classification: ResNet‑18 on CIFAR‑10 (PyTorch)

Create `src/classification/train_resnet_cifar10.py` with:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3

# CIFAR-10: 32x32 RGB, 10 classes
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_ds = datasets.CIFAR10(root="datasets", train=True, download=True, transform=transform_train)
test_ds  = datasets.CIFAR10(root="datasets", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Adjust classifier head for 10 classes
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

best_acc = 0.0
save_dir = Path("models"); save_dir.mkdir(parents=True, exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    # Eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    test_acc = correct / total

    print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | test_acc={test_acc:.4f}")

    # Save best
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), save_dir / "resnet18_cifar10.pt")
        print(f"Saved best model with acc={best_acc:.4f}")
```

Run it:

```bash
python src/classification/train_resnet_cifar10.py
```

The best checkpoint is saved to `models/resnet18_cifar10.pt`.

---

### Streamlit image detector app

Create `apps/streamlit_yolo_app/app.py` with:

```python
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

st.set_page_config(page_title="Vision-Lab • YOLOv8 Image Detector")
st.title("YOLOv8 Image Detector")
st.caption("Upload an image to run object detection with a pretrained YOLOv8 model.")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_column_width=True)

    results = model.predict(np.array(img))
    annotated = results[0].plot()
    st.image(annotated, caption="Detections", use_column_width=True)
```

Run it:

```bash
streamlit run apps/streamlit_yolo_app/app.py
```

---

## Datasets

* **COCO128** (Ultralytics sample): tiny subset for fast detection training/eval — auto‑referenced by `data=coco128.yaml` in YOLO commands.
* **COCO128‑Seg**: tiny subset for segmentation — `data=coco128-seg.yaml`.
* **CIFAR‑10**: auto‑downloaded by the PyTorch script into `datasets/`.

> Larger datasets (COCO, Pascal VOC, Cityscapes) can be added later. Start small to validate pipelines quickly.

---

## Reproducibility

* Set seeds in your training scripts to stabilize results across runs.
* Keep environment details (`pip freeze > docs/requirements-lock.txt`) next to experiment reports.
* Log key metrics (accuracy, mAP, IoU) in README tables per project as you iterate.

```python
import torch, random, numpy as np
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

---

## Performance Notes

* **Model choice**: `yolov8n` is fast & light for CPU/webcam demos; `yolov8s/m/l/x` trade speed for accuracy.
* **Batching**: Use higher batch sizes on GPU for faster training.
* **Augmentations**: Try Albumentations for stronger generalization on custom datasets.
* **Export**: Ultralytics supports ONNX, TensorRT, CoreML for deployment beyond Python.

---

## License

MIT License

---

## Author

**Vishesh Srivastava**
Website: [https://visheshsrivastava.com/](https://visheshsrivastava.com/)
LinkedIn: [https://linkedin.com/in/iamvisheshsrivastava](https://linkedin.com/in/iamvisheshsrivastava)
GitHub: [https://github.com/iamvisheshsrivastava](https://github.com/iamvisheshsrivastava)
