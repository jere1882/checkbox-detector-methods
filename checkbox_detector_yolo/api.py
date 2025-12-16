"""
FastAPI endpoint for checkbox detection using YOLO model
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import os
from typing import Optional
import tempfile

app = FastAPI(
    title="Checkbox Detection API",
    description="YOLOv11-based checkbox detection service",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
MODEL_PATH = "runs/detect/train/weights/best.pt"

@app.on_event("startup")
async def load_model():
    """Load the YOLO model on startup"""
    global model
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠ Warning: Model weights not found at {MODEL_PATH}")
        print("  Please ensure model weights are available before making predictions")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Checkbox Detection API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "predict_image": "/predict/image (POST)"
        }
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.2):
    """
    Predict checkboxes in an uploaded image
    
    Args:
        file: Image file (jpg, png, etc.)
        conf: Confidence threshold (default: 0.2)
    
    Returns:
        JSON with detection results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Run inference
        results = model.predict(image, conf=conf, verbose=False)
        result = results[0]
        boxes = result.boxes
        
        # Extract detections
        detections = []
        class_names = {0: "empty_checkbox", 1: "filled_checkbox"}
        
        if len(boxes) > 0:
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf_score = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                detections.append({
                    "class": class_names[cls_id],
                    "confidence": round(conf_score, 3),
                    "bbox": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3])
                    }
                })
        
        # Count by class
        counts = {name: sum(1 for d in detections if d["class"] == name) 
                 for name in class_names.values()}
        
        return JSONResponse({
            "success": True,
            "detections": detections,
            "counts": counts,
            "total": len(detections)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...), conf: float = 0.2):
    """
    Predict checkboxes and return annotated image
    
    Args:
        file: Image file (jpg, png, etc.)
        conf: Confidence threshold (default: 0.2)
    
    Returns:
        Annotated image with bounding boxes
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Run inference
        results = model.predict(image, conf=conf, verbose=False)
        result = results[0]
        
        # Get annotated image
        annotated_image = result.plot()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            cv2.imwrite(tmp_file.name, annotated_image)
            tmp_path = tmp_file.name
        
        return FileResponse(
            tmp_path,
            media_type="image/jpeg",
            filename="detected_checkboxes.jpg"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




