"""
Lambda handler for FastAPI application using Mangum
"""
from mangum import Mangum
import os
import shutil

# Copy model to /tmp (Lambda's writable directory) before importing api
MODEL_SRC = "/var/task/runs/detect/train/weights/best.pt"
MODEL_DST = "/tmp/best.pt"

if os.path.exists(MODEL_SRC) and not os.path.exists(MODEL_DST):
    print(f"Copying model to {MODEL_DST}")
    os.makedirs(os.path.dirname(MODEL_DST), exist_ok=True)
    shutil.copy2(MODEL_SRC, MODEL_DST)
    print(f"✓ Model copied to {MODEL_DST}")

# Set environment variable for model path before importing api
os.environ["MODEL_PATH"] = MODEL_DST

# Now import api and load model
from api import app
import api
from ultralytics import YOLO

# Load model if not already loaded
if api.model is None and os.path.exists(MODEL_DST):
    print(f"Loading model from {MODEL_DST}")
    api.model = YOLO(MODEL_DST)
    print(f"✓ Model loaded successfully")

# Create Mangum handler to wrap FastAPI for Lambda
handler = Mangum(app, lifespan="off")

# Lambda will call this handler function
def lambda_handler(event, context):
    return handler(event, context)

