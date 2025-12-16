# Repository Architecture

This document explains the structure and design decisions of the checkbox detection repository.

## Overview

This repository compares **three approaches** to checkbox detection in scanned documents:

1. **OpenCV** — Traditional computer vision (contour detection, heuristics)
2. **YOLOv11** — Deep learning object detection (transfer learning)
3. **Gemini** — Vision LLM prompting (zero-shot, API-based)

Each approach lives in its own self-contained folder, making it easy to understand, test, and deploy independently.

## Folder Structure

```
checkbox_detector/
├── data/                         # Shared dataset
│   ├── train/                    # Training images + labels
│   │   ├── images/
│   │   └── labels/
│   └── val/                      # Validation images + labels
│       ├── images/
│       └── labels/
│
├── checkbox_detector_opencv/     # Approach 1: Traditional CV
│   ├── checkbox_detector/        # Python package
│   ├── scripts/                  # CLI scripts
│   ├── tests/                    # Unit tests
│   └── data/                     # Sample images (for demos)
│
├── checkbox_detector_yolo/       # Approach 2: Deep Learning
│   ├── notebooks/                # Training + inference notebooks
│   ├── runs/                     # Model outputs (gitignored)
│   ├── api.py                    # FastAPI REST API
│   ├── test_inference.py         # CLI inference script
│   ├── docker-compose.yml        # Container orchestration
│   ├── Dockerfile                # Container definition
│   └── data.yaml                 # Dataset config (points to ../data/)
│
├── checkbox_detector_gemini/     # Approach 3: Vision LLM
│   ├── detect_checkboxes.py      # Main detection script
│   ├── requirements.txt          # Dependencies
│   └── README.md                 # Usage instructions
│
├── docs/                         # Documentation images
├── .github/workflows/            # CI/CD pipelines
├── .env                          # API keys (gitignored)
└── README.md                     # Main project README
```

## Why This Structure?

### A. Folder Organization Rationale

| Design Choice | Reason |
|---------------|--------|
| **Separate folders per approach** | Each implementation is self-contained. You can delete `checkbox_detector_opencv/` without affecting YOLO. Makes comparison fair and clear. |
| **Shared `/data` at root** | Allows all approaches to use the same dataset. Enables apples-to-apples comparison. Easy to expand with more training data. |
| **Each approach has its own README** | Users can dive into one approach without reading about others. |
| **Docker files inside YOLO folder** | YOLO is the only approach that benefits from containerization (heavy dependencies). Keeps it self-contained. |

### B. Why Not a Monolithic Package?

A single `checkbox_detector` package with submodules (e.g., `checkbox_detector.opencv`, `checkbox_detector.yolo`) would:
- Force users to install all dependencies (PyTorch, ultralytics, OpenCV, google-genai)
- Make the codebase harder to navigate
- Complicate Docker builds

The current structure lets you:
```bash
# Only use OpenCV approach
cd checkbox_detector_opencv && pip install -e .

# Only use YOLO approach  
cd checkbox_detector_yolo && pip install -r requirements.txt

# Only use Gemini approach
cd checkbox_detector_gemini && pip install -r requirements.txt
```

---

## Containerization (Docker)

### Which Approaches Use Docker?

| Approach | Dockerized? | Reason |
|----------|-------------|--------|
| **OpenCV** | ❌ No | Lightweight, pure Python. `pip install` is sufficient. |
| **YOLO** | ✅ Yes | Heavy dependencies (PyTorch, CUDA). Docker ensures reproducibility. |
| **Gemini** | ❌ No | API-based. Only needs `google-genai` package. |

### Why Docker for YOLO?

1. **Reproducibility**: PyTorch + CUDA versions must match exactly. Docker freezes the environment.
2. **Deployment**: The FastAPI service can be deployed anywhere Docker runs.
3. **Isolation**: Doesn't pollute your system Python with ML libraries.
4. **GPU Support**: NVIDIA Container Toolkit enables GPU passthrough.

### Docker Files

```
checkbox_detector_yolo/
├── Dockerfile           # Builds the inference service image
└── docker-compose.yml   # Orchestrates the service
```

#### Dockerfile Overview

```dockerfile
FROM python:3.9-slim                    # Lightweight base
RUN apt-get install libgl1-mesa-glx...  # OpenCV system deps
COPY requirements.txt .
RUN pip install -r requirements.txt     # Install dependencies
COPY . .                                # Copy app code
EXPOSE 8000                             # FastAPI port
CMD ["uvicorn", "api:app", ...]         # Start API server
```

#### docker-compose.yml Overview

```yaml
services:
  checkbox-detector-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./runs/detect/train/weights:/app/runs/detect/train/weights:ro  # Model weights
      - ../data:/app/data:ro                                            # Dataset access
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
```

### How to Run

```bash
cd checkbox_detector_yolo

# Build and start the service
docker-compose up -d

# Check health
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true}

# Run inference
curl -X POST "http://localhost:8000/predict" \
  -F "file=@../data/val/images/real.jpg" \
  | jq

# Stop
docker-compose down
```

### Advantages of Containerization

| Advantage | Description |
|-----------|-------------|
| **Portability** | Runs on any machine with Docker (Linux, Mac, Windows WSL2) |
| **Reproducibility** | Same environment in dev, CI, and production |
| **Isolation** | No dependency conflicts with other projects |
| **Easy deployment** | Push to registry → `docker pull` → `docker run` |
| **Scalability** | Easy to scale horizontally with Kubernetes/Docker Swarm |

---

## YOLO: GPU Requirements and Weight Loading

### Do You Need a GPU?

| Task | GPU Required? | Notes |
|------|---------------|-------|
| **Training** | ✅ Recommended | Training on CPU is extremely slow (hours vs minutes) |
| **Inference** | ❌ Optional | CPU inference works fine (~250ms per image) |
| **Docker API** | ❌ Optional | Falls back to CPU automatically |

#### GPU Inference (if available)

The YOLO library auto-detects CUDA:
```python
from ultralytics import YOLO
model = YOLO("best.pt")
# Automatically uses GPU if available, CPU otherwise
results = model.predict("image.jpg")
```

#### Force CPU

```python
results = model.predict("image.jpg", device="cpu")
```

### How Weights Are Loaded

#### 1. Training Produces Weights

After training (in `notebooks/train_and_visualize_prototype.ipynb`), weights are saved:
```
runs/detect/train/weights/
├── best.pt    # Best checkpoint (lowest val loss)
└── last.pt    # Final epoch checkpoint
```

#### 2. Inference Loads Weights

**CLI Script (`test_inference.py`):**
```python
from ultralytics import YOLO

# Default path
model = YOLO("runs/detect/train/weights/best.pt")

# Or custom path via --weights flag
model = YOLO(args.weights)
```

**FastAPI (`api.py`):**
```python
MODEL_PATH = "runs/detect/train/weights/best.pt"

@app.on_event("startup")
async def load_model():
    global model
    model = YOLO(MODEL_PATH)  # Load once at startup
```

**Docker:**
Weights are mounted as a volume, so you can update them without rebuilding:
```yaml
volumes:
  - ./runs/detect/train/weights:/app/runs/detect/train/weights:ro
```

### Weight File Format

- **Format**: PyTorch checkpoint (`.pt`)
- **Size**: ~6MB (YOLOv11n backbone)
- **Contents**: Model architecture + trained parameters
- **Portability**: Works on any machine with `ultralytics` installed

---

## Gemini: Vision LLM Approach

The Gemini approach uses Google's multimodal LLM to detect checkboxes via prompting. No training required.

### How It Works

1. Load image as bytes
2. Send to Gemini 2.5 Pro with a structured prompt
3. Gemini returns bounding boxes in JSON format
4. Parse and visualize results

### Advantages

- **Zero training data needed**
- **Zero GPU needed** (API-based)
- **Highly flexible** (just change the prompt)

### Disadvantages

- **API cost** (~$0.01-0.05 per image)
- **Latency** (~2-5 seconds per image)
- **Rate limits**
- **Less consistent** than trained models

---

## Summary

| Aspect | OpenCV | YOLO | Gemini |
|--------|--------|------|--------|
| **Dependencies** | Lightweight | Heavy (PyTorch) | Minimal (API client) |
| **Docker** | No | Yes | No |
| **GPU** | No | Optional (recommended for training) | No (API-based) |
| **Training Data** | None | Required | None |
| **Latency** | ~10ms | ~50-250ms | ~2-5s |
| **Accuracy** | Lower | Higher | Variable |

This architecture enables fair comparison while keeping each approach production-ready and independently deployable.


