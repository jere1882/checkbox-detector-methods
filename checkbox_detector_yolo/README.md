# Checkbox Detection using YOLOv11

This repository contains a YOLOv11-based object detection model for detecting and classifying checkboxes (filled vs. unfilled) in scanned documents.

## Quick Start

### Inference

You can run inference using multiple methods:

**Option 1: Docker API (Recommended for production)**

```bash
# From this directory (checkbox_detector_yolo/)
docker-compose up -d

# Test the API
curl http://localhost:8000/health

# Use the API (see API Usage section below)
```

**Option 2: Command-line script**

```bash
python test_inference.py input_image.jpg output_image.jpg
```

Options:
- `input_image`: Path to input image (required)
- `output_image`: Path to save annotated output image (optional)
- `--weights`: Path to model weights (default: `runs/detect/train/weights/best.pt`)
- `--conf`: Confidence threshold (default: 0.2)

**Option 3: Jupyter Notebook**

```bash
jupyter notebook notebooks/inference.ipynb
```

**Option 4: Python API directly**

```bash
pip install -r requirements.txt
python api.py
# API available at http://localhost:8000
```

The trained model weights are located at `runs/detect/train/weights/best.pt`.

### Training

To train the model from scratch or retrain:

```bash
jupyter notebook notebooks/train_and_visualize_prototype.ipynb
```

For results and discussion, see the main training notebook: `notebooks/train_and_visualize_prototype.ipynb`

# Set Up your environment

Create a conda environment

```bash
conda create -n yolov11-env python=3.9 -y
conda activate yolov11-env
```

Install / Update Pytorch as needed by following the [official instructions](https://pytorch.org/get-started/locally/), e.g:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install other packages:

```bash
pip install -r requirements.txt
```

If you encounter problems installing ultralytics, refer to the [official instructions](https://docs.ultralytics.com/quickstart/).

## Project Structure

```
checkbox_detector_yolo/
├── notebooks/                    # Jupyter notebooks
│   ├── train_and_visualize_prototype.ipynb  # Main training notebook
│   ├── inference.ipynb           # Inference notebook
│   └── train_and_visualize_prototype.md     # Exported notebook
├── runs/                        # Training outputs (gitignored)
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt      # Trained model weights
├── docs/                        # Documentation images
├── api.py                       # FastAPI REST API endpoint
├── test_inference.py            # Command-line inference script
├── docker-compose.yml           # Docker Compose configuration
├── Dockerfile                   # Docker container definition
├── data.yaml                    # Dataset configuration (points to ../data/)
└── requirements.txt             # Python dependencies

# Dataset is shared at repository root:
../data/
├── train/                       # Training images and labels
│   ├── images/
│   └── labels/
└── val/                         # Validation images and labels
    ├── images/
    └── labels/
```

## Dataset

The model was trained on just **5 annotated images** with two classes:
- `empty_checkbox` (class 0)
- `filled_checkbox` (class 1)

Despite the minimal dataset, the model achieves good generalization thanks to transfer learning from YOLOv11's pretrained weights.

# Troubleshooting

If your jupyter instance is finding trouble getting linked to the conda env, try:

```bash
conda install ipykernel
python -m ipykernel install --user --name=yolov11 --display-name "Python yolov11"
```

And then select kernel yolov11 when you run the notebooks.

## Deployment

### Docker Compose (Recommended)

The easiest way to deploy the API service:

```bash
# From this directory (checkbox_detector_yolo/)
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop the service
docker-compose down
```

The API will be available at `http://localhost:8000`

### Building Docker Image Manually

```bash
docker build -t checkbox-detector:latest .
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/runs/detect/train/weights:/app/runs/detect/train/weights:ro \
  --name checkbox-detector \
  checkbox-detector:latest
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Predict Checkboxes (JSON Response)

```bash
curl -X POST "http://localhost:8000/predict?conf=0.2" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

### Predict Checkboxes (Annotated Image)

```bash
curl -X POST "http://localhost:8000/predict/image?conf=0.2" \
  -H "accept: image/jpeg" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg" \
  --output detected_checkboxes.jpg
```

### Python Client Example

```python
import requests

# Predict checkboxes
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f},
        params={'conf': 0.2}
    )
    results = response.json()
    print(f"Detected {results['total']} checkboxes")
    print(f"Empty: {results['counts']['empty_checkbox']}")
    print(f"Filled: {results['counts']['filled_checkbox']}")

# Get annotated image
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict/image',
        files={'file': f}
    )
    with open('output.jpg', 'wb') as out:
        out.write(response.content)
```

### Interactive API Documentation

Once the service is running, visit:
- `http://localhost:8000/docs` - Swagger UI
- `http://localhost:8000/redoc` - ReDoc

## Production Considerations

For production deployment, consider:

1. **Model Weights**: Use a model registry or object storage (S3, GCS) instead of mounting local files
2. **Scaling**: Use Kubernetes or Docker Swarm for horizontal scaling
3. **Monitoring**: Add Prometheus metrics and logging (e.g., ELK stack)
4. **Security**: 
   - Use HTTPS/TLS
   - Add authentication/authorization
   - Rate limiting
   - Input validation and file size limits
5. **Resource Limits**: Set appropriate CPU/memory limits in Docker
6. **Health Checks**: The container includes a health check endpoint

## Troubleshooting

### Model Not Loading

If you see "Model not loaded" errors:
1. Verify model weights exist at `runs/detect/train/weights/best.pt`
2. Check Docker volume mounts are correct
3. Review container logs: `docker-compose logs`

### Port Already in Use

Change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Use port 8001 instead
```

### GPU Support

For GPU acceleration, modify the Dockerfile to use `nvidia/cuda` base image and add GPU runtime flags to docker-compose.yml.
