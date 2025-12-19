# Lambda Deployment Walkthrough: From Zero to Production

This document summarizes the complete process of deploying the YOLO checkbox detector to AWS Lambda, including all design choices, challenges faced, and solutions implemented.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites & Credentials](#prerequisites--credentials)
3. [Architecture Choices](#architecture-choices)
4. [Step-by-Step Deployment](#step-by-step-deployment)
5. [The API Gateway Struggle](#the-api-gateway-struggle)
6. [Lambda Function URL vs API Gateway](#lambda-function-url-vs-api-gateway)
7. [Final Working Solution](#final-working-solution)
8. [Libraries Involved](#libraries-involved)
9. [Lessons Learned](#lessons-learned)

---

## Overview

**Goal:** Deploy a custom-trained YOLOv11 checkbox detection model as a serverless API on AWS Lambda.

**Stack:**
- Python 3.9
- FastAPI (web framework)
- Mangum (FastAPI → Lambda adapter)
- Docker (containerization)
- AWS Lambda (serverless compute)
- AWS ECR (container registry)
- AWS API Gateway (HTTP endpoint)
- AWS Lambda Function URL (alternative HTTP endpoint)

**Final Result:**
- Lambda function: `checkbox-detector`
- API Gateway URL: `https://0q9xmrf4ue.execute-api.us-east-2.amazonaws.com/prod`
- Lambda Function URL: `https://t6lwvtxko5lg6txdxp5b5gs2ve0mtusz.lambda-url.us-east-2.on.aws/`

---

## Prerequisites & Credentials

### AWS Credentials Required

We used an existing AWS CLI profile called `egyptian-project`:

```bash
# Check available profiles
aws configure list-profiles

# Verify the profile works
aws sts get-caller-identity --profile egyptian-project
# Output: Account: 342593762849, User: jrodriguez
```

**Required IAM Permissions:**
- `ecr:CreateRepository`, `ecr:GetAuthorizationToken`, `ecr:BatchCheckLayerAvailability`, `ecr:PutImage`
- `lambda:CreateFunction`, `lambda:UpdateFunctionCode`, `lambda:UpdateFunctionConfiguration`, `lambda:GetFunction`, `lambda:AddPermission`, `lambda:CreateFunctionUrlConfig`
- `iam:CreateRole`, `iam:AttachRolePolicy`, `iam:GetRole`
- `apigateway:*` (for API Gateway setup)

### Local Requirements

- Docker installed and running
- User in `docker` group (to run without sudo)
- AWS CLI configured
- Model weights at `runs/detect/train/weights/best.pt`

---

## Architecture Choices

### Why Lambda?

1. **Serverless**: No servers to manage
2. **Pay-per-use**: Only pay when invoked
3. **Auto-scaling**: Handles traffic spikes automatically
4. **Already familiar**: User had deployed similar projects (Egyptian Art Analyzer)

### Why Docker Container (not ZIP)?

1. **Size**: YOLO + PyTorch + OpenCV exceeds Lambda's 250MB ZIP limit
2. **Dependencies**: Complex native libraries (OpenCV, PyTorch) are easier in Docker
3. **Reproducibility**: Same environment locally and in Lambda

### Why FastAPI + Mangum?

1. **FastAPI**: Modern, async Python framework with automatic OpenAPI docs
2. **Mangum**: Adapter that converts Lambda events to ASGI (FastAPI's protocol)
3. **Reusability**: Same `api.py` works locally with uvicorn and in Lambda with Mangum

### Lambda Configuration

```python
MEMORY_SIZE = 3008  # MB (Lambda max for optimal performance)
TIMEOUT = 900       # seconds (15 minutes, Lambda max)
```

Why these values?
- **Memory**: ML models are memory-hungry. More memory also means more CPU.
- **Timeout**: Model loading can take 30-60 seconds on cold start.

---

## Step-by-Step Deployment

### Step 1: Create Lambda Handler

The key file that bridges FastAPI and Lambda:

```python
# lambda_handler.py
from mangum import Mangum
import os
import shutil

# Copy model to /tmp (Lambda's writable directory)
MODEL_SRC = "/var/task/runs/detect/train/weights/best.pt"
MODEL_DST = "/tmp/best.pt"

if os.path.exists(MODEL_SRC) and not os.path.exists(MODEL_DST):
    shutil.copy2(MODEL_SRC, MODEL_DST)

# Load model
from api import app
import api
from ultralytics import YOLO

if api.model is None and os.path.exists(MODEL_DST):
    api.model = YOLO(MODEL_DST)

# Wrap FastAPI for Lambda
handler = Mangum(app, lifespan="off")

def lambda_handler(event, context):
    return handler(event, context)
```

**Key decisions:**
- `lifespan="off"`: Disables FastAPI startup events (they don't work in Lambda)
- Model copied to `/tmp`: Lambda's `/var/task` is read-only
- Model loaded at import time: Ensures it's ready before first request

### Step 2: Create Lambda-Compatible Dockerfile

```dockerfile
FROM public.ecr.aws/lambda/python:3.9

WORKDIR ${LAMBDA_TASK_ROOT}

# Install OpenCV dependencies (Amazon Linux uses yum)
RUN yum install -y mesa-libGL glib2 && yum clean all

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn python-multipart pillow mangum

# Copy application code
COPY api.py .
COPY lambda-deployment/lambda_handler.py lambda_handler.py

# Copy model weights with proper permissions
RUN mkdir -p ${LAMBDA_TASK_ROOT}/runs/detect/train/weights
COPY runs/detect/train/weights/best.pt ${LAMBDA_TASK_ROOT}/runs/detect/train/weights/best.pt
RUN chmod 644 ${LAMBDA_TASK_ROOT}/runs/detect/train/weights/best.pt

# Set Lambda handler
CMD [ "lambda_handler.lambda_handler" ]
```

**Key decisions:**
- Base image: `public.ecr.aws/lambda/python:3.9` (official Lambda image)
- `chmod 644`: Ensures model file is readable (critical fix!)
- Model baked into image: Faster cold starts vs downloading from S3

### Step 3: Build and Push Docker Image

```bash
# Build image
docker build -f Dockerfile.lambda -t checkbox-detector:latest .

# Create ECR repository
aws ecr create-repository --profile egyptian-project \
  --repository-name checkbox-detector --region us-east-2

# Login to ECR
aws ecr get-login-password --profile egyptian-project --region us-east-2 | \
  docker login --username AWS --password-stdin \
  342593762849.dkr.ecr.us-east-2.amazonaws.com

# Tag and push
docker tag checkbox-detector:latest \
  342593762849.dkr.ecr.us-east-2.amazonaws.com/checkbox-detector:latest
docker push 342593762849.dkr.ecr.us-east-2.amazonaws.com/checkbox-detector:latest
```

### Step 4: Create Lambda Function

```bash
# Create IAM role for Lambda
aws iam create-role --profile egyptian-project \
  --role-name lambda-execution-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach basic execution policy
aws iam attach-role-policy --profile egyptian-project \
  --role-name lambda-execution-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Create Lambda function
aws lambda create-function --profile egyptian-project \
  --function-name checkbox-detector \
  --package-type Image \
  --code ImageUri=342593762849.dkr.ecr.us-east-2.amazonaws.com/checkbox-detector:latest \
  --timeout 900 \
  --memory-size 3008 \
  --role arn:aws:iam::342593762849:role/lambda-execution-role \
  --region us-east-2
```

### Step 5: Create API Gateway

```bash
# Create REST API
API_ID=$(aws apigateway create-rest-api --profile egyptian-project \
  --name checkbox-detector-API --region us-east-2 \
  --query 'id' --output text)

# Get root resource
ROOT_ID=$(aws apigateway get-resources --profile egyptian-project \
  --rest-api-id $API_ID --region us-east-2 \
  --query 'items[?path==`/`].id' --output text)

# Create proxy resource {proxy+}
PROXY_ID=$(aws apigateway create-resource --profile egyptian-project \
  --rest-api-id $API_ID --parent-id $ROOT_ID \
  --path-part '{proxy+}' --region us-east-2 \
  --query 'id' --output text)

# Create ANY method with Lambda integration
aws apigateway put-method --profile egyptian-project \
  --rest-api-id $API_ID --resource-id $PROXY_ID \
  --http-method ANY --authorization-type NONE --region us-east-2

aws apigateway put-integration --profile egyptian-project \
  --rest-api-id $API_ID --resource-id $PROXY_ID \
  --http-method ANY --type AWS_PROXY \
  --integration-http-method POST \
  --uri "arn:aws:apigateway:us-east-2:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-2:342593762849:function:checkbox-detector/invocations" \
  --region us-east-2

# Add Lambda permission for API Gateway
aws lambda add-permission --profile egyptian-project \
  --function-name checkbox-detector \
  --statement-id apigateway-invoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:us-east-2:342593762849:$API_ID/*/*" \
  --region us-east-2

# Deploy API
aws apigateway create-deployment --profile egyptian-project \
  --rest-api-id $API_ID --stage-name prod --region us-east-2
```

---

## The API Gateway Struggle

### The Problem

After deploying, the `/health` endpoint worked, but `/predict` with file uploads failed:

```bash
curl -X POST https://API_GATEWAY_URL/predict -F "file=@image.jpg"
# {"detail":"Prediction failed: 400: Could not decode image"}
```

### Why It Failed

**API Gateway has issues with binary/multipart data:**

1. **Multipart form encoding**: When you send `-F "file=@image.jpg"`, curl sends a `multipart/form-data` request
2. **API Gateway transformation**: API Gateway converts the request to Lambda's event format
3. **Binary corruption**: Multipart binary data can get corrupted in this transformation
4. **Base64 encoding**: API Gateway may base64-encode binary data, but the headers/boundaries get mangled

### How the Egyptian Art Analyzer Avoided This

Looking at your Egyptian project's `lambda_function.py`:

```python
# Egyptian project receives JSON with base64 image
request_data = json.loads(body)
image_data = request_data.get('image')  # base64 string
```

Your Egyptian project sends images as **base64-encoded strings inside JSON**, not as multipart file uploads. JSON is text-based and API Gateway handles it perfectly.

### The Solution

Added a new endpoint that accepts base64 JSON (like your Egyptian project):

```python
class PredictRequest(BaseModel):
    image: str  # base64 encoded image
    conf: float = 0.2

@app.post("/predict/json")
async def predict_json(request: PredictRequest):
    # Decode base64 image
    image_data = base64.b64decode(request.image)
    # ... rest of prediction logic
```

Now API Gateway works:

```bash
# Encode image to base64
IMAGE_BASE64=$(base64 -w 0 image.jpg)

# Send as JSON
curl -X POST https://API_GATEWAY_URL/predict/json \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE_BASE64\", \"conf\": 0.2}"
# Works!
```

---

## Lambda Function URL vs API Gateway

### What is Lambda Function URL?

Lambda Function URL is a simpler alternative to API Gateway. It provides a direct HTTPS endpoint to your Lambda function without the API Gateway layer.

```bash
# Create Function URL
aws lambda create-function-url-config \
  --function-name checkbox-detector \
  --auth-type NONE \
  --profile egyptian-project --region us-east-2

# Output: https://t6lwvtxko5lg6txdxp5b5gs2ve0mtusz.lambda-url.us-east-2.on.aws/

# Allow public access
aws lambda add-permission \
  --function-name checkbox-detector \
  --statement-id FunctionURLAllowPublicAccess \
  --action lambda:InvokeFunctionUrl \
  --principal "*" \
  --function-url-auth-type NONE \
  --profile egyptian-project --region us-east-2
```

### Comparison

| Feature | API Gateway | Lambda Function URL |
|---------|-------------|---------------------|
| **Binary handling** | Tricky (requires config) | Native support |
| **Multipart forms** | Problematic | Works perfectly |
| **Pricing** | $3.50/million requests + data | Free (included in Lambda) |
| **Features** | Rate limiting, API keys, caching, transforms | Basic HTTPS endpoint |
| **Custom domains** | Yes | Yes (via CloudFront) |
| **WebSocket** | Yes | No |
| **Request validation** | Yes | No |

### Why Function URL Worked for Multipart

Lambda Function URL passes the request more directly to Lambda:
- Binary data stays intact
- Multipart boundaries are preserved
- No transformation layer

### Recommendation

- **Use Function URL** for simple APIs, file uploads, internal services
- **Use API Gateway** for complex APIs needing rate limiting, API keys, request validation, caching

---

## Final Working Solution

### Two Ways to Call the API

#### 1. Lambda Function URL (Multipart - simpler for curl/scripts)

```bash
# Health check
curl https://t6lwvtxko5lg6txdxp5b5gs2ve0mtusz.lambda-url.us-east-2.on.aws/health

# Predict with file upload
curl -X POST https://t6lwvtxko5lg6txdxp5b5gs2ve0mtusz.lambda-url.us-east-2.on.aws/predict \
  -F "file=@image.jpg" \
  -F "conf=0.2"
```

#### 2. API Gateway (Base64 JSON - like Egyptian project)

```bash
# Health check
curl https://0q9xmrf4ue.execute-api.us-east-2.amazonaws.com/prod/health

# Predict with base64 JSON
IMAGE_BASE64=$(base64 -w 0 image.jpg)
echo "{\"image\": \"$IMAGE_BASE64\", \"conf\": 0.2}" > payload.json

curl -X POST https://0q9xmrf4ue.execute-api.us-east-2.amazonaws.com/prod/predict/json \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### Available Endpoints

| Endpoint | Method | Description | Works via |
|----------|--------|-------------|-----------|
| `/` | GET | API info | Both |
| `/health` | GET | Health check | Both |
| `/predict` | POST | Multipart file upload | Function URL only |
| `/predict/json` | POST | Base64 JSON | Both (API Gateway compatible) |
| `/predict/image` | POST | Returns annotated image | Function URL only |

### Test Results

```
Success: True
Total detections: 47
Empty checkboxes: 25
Filled checkboxes: 22
```

---

## Libraries Involved

### Python Dependencies

| Library | Purpose |
|---------|---------|
| `ultralytics` | YOLOv11 model inference |
| `torch` | PyTorch (neural network runtime) |
| `opencv-python` | Image processing |
| `fastapi` | Web framework |
| `mangum` | FastAPI → Lambda adapter |
| `uvicorn` | ASGI server (local development) |
| `pillow` | Image handling |
| `python-multipart` | Multipart form parsing |
| `pydantic` | Request validation |

### AWS Services

| Service | Purpose |
|---------|---------|
| Lambda | Serverless compute |
| ECR | Docker image registry |
| API Gateway | REST API endpoint |
| IAM | Permissions and roles |
| CloudWatch | Logs |

---

## Lessons Learned

### 1. Lambda's `/var/task` is Read-Only

**Problem:** YOLO tried to write to the model file, causing permission errors.

**Solution:** Copy model to `/tmp` (Lambda's writable directory) before loading:
```python
shutil.copy2("/var/task/model.pt", "/tmp/model.pt")
model = YOLO("/tmp/model.pt")
```

### 2. File Permissions Matter

**Problem:** Model file copied to Docker image wasn't readable.

**Solution:** Add `chmod` in Dockerfile:
```dockerfile
RUN chmod 644 ${LAMBDA_TASK_ROOT}/runs/detect/train/weights/best.pt
```

### 3. Mangum's `lifespan="off"` Skips Startup Events

**Problem:** FastAPI's `@app.on_event("startup")` doesn't run in Lambda with Mangum.

**Solution:** Load model at module import time in `lambda_handler.py`, not in startup event.

### 4. API Gateway + Binary Data = Pain

**Problem:** Multipart file uploads fail through API Gateway.

**Solutions:**
- Use Lambda Function URL for file uploads
- Or send images as base64 JSON (like Egyptian project)

### 5. Cold Starts Are Slow for ML

**Problem:** First request takes 30-60 seconds (loading PyTorch + model).

**Mitigations:**
- Provisioned concurrency (keeps instances warm)
- Smaller model (YOLOv11n instead of larger variants)
- Consider Cloud Run for better cold start handling

### 6. Always Test with Real Data

**Problem:** Health endpoint worked, but prediction failed silently.

**Solution:** Always test the actual use case (image prediction) end-to-end.

---

## Quick Reference: Key Commands

```bash
# Check Lambda status
aws lambda get-function --function-name checkbox-detector \
  --profile egyptian-project --region us-east-2 \
  --query 'Configuration.{State:State,LastUpdate:LastUpdateStatus}'

# View logs
aws logs tail /aws/lambda/checkbox-detector \
  --profile egyptian-project --region us-east-2 --follow

# Update function (after code changes)
cd lambda-deployment && bash deploy_lambda.sh

# Test health
curl https://t6lwvtxko5lg6txdxp5b5gs2ve0mtusz.lambda-url.us-east-2.on.aws/health

# Test prediction
curl -X POST https://t6lwvtxko5lg6txdxp5b5gs2ve0mtusz.lambda-url.us-east-2.on.aws/predict \
  -F "file=@image.jpg"
```

---

## Files Created

```
lambda-deployment/
├── deploy_lambda.sh          # Main deployment script
├── setup_api_gateway.sh      # API Gateway setup
├── Dockerfile.lambda         # Lambda-compatible Docker image
├── lambda_handler.py         # Mangum wrapper for FastAPI
├── LAMBDA_DEPLOYMENT.md      # Deployment guide
└── DEPLOYMENT_WALKTHROUGH_SUMMARY.md  # This file
```

---

## Appendix A: What Does "Docker Running" Mean?

Docker has two parts:

### 1. Docker CLI (the command)

The `docker` command you type in the terminal. This is just a client that sends commands to the daemon.

### 2. Docker Daemon (the service)

A background process (`dockerd`) that actually:
- Builds images
- Runs containers
- Manages networks and volumes
- Stores images locally

### "Docker running" means the daemon is active

```bash
# Check if Docker daemon is running
docker ps
# If it works → daemon is running
# If "Cannot connect to the Docker daemon" → daemon is NOT running

# Or check the service directly (Linux)
sudo systemctl status docker
```

### What happens if the daemon isn't running?

```bash
docker build -t my-image .
# Error: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. 
# Is the docker daemon running?
```

### How to start it

```bash
# Linux (systemd)
sudo systemctl start docker

# macOS/Windows
# Open Docker Desktop app - it starts the daemon automatically

# Enable auto-start on boot (Linux)
sudo systemctl enable docker
```

### Why it's a "daemon"

Docker runs as a background service because:
1. **Persistent state**: Keeps track of running containers, images, networks
2. **Resource management**: Manages CPU, memory allocation to containers
3. **Always available**: Other processes can connect to it anytime

Think of it like a database server - MySQL or PostgreSQL also run as daemons. You need the server running before you can connect to it.

---

## Appendix B: Why FastAPI + Mangum Instead of Direct Lambda Handler?

### The Question

Why did we use FastAPI + Mangum when we could have written a simple `lambda_handler` function directly, like the Egyptian Art Analyzer project?

### The Honest Answer

**We didn't need FastAPI.** The `api.py` already existed for Docker deployment, and we reused it rather than rewriting. This was pragmatic, not necessary.

### Direct Lambda Handler (Simpler Approach)

The Egyptian project does it the simpler way:

```python
# Direct Lambda handler - no framework
def lambda_handler(event, context):
    body = json.loads(event.get('body', '{}'))
    image_data = body.get('image')
    
    result = process_image(image_data)
    
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(result)
    }
```

We could have done the same (~50 lines instead of ~270 lines + Mangum).

### When FastAPI Actually Helps

| Feature | Useful For |
|---------|------------|
| **Local development** | Run `uvicorn api:app --reload` for fast iteration |
| **Automatic docs** | Visit `/docs` for Swagger UI |
| **Request validation** | Pydantic validates inputs, returns nice errors |
| **Multiple endpoints** | Complex APIs with many routes |
| **Portability** | Same code runs in Docker, Lambda, or bare metal |

### When Direct Lambda Handler Is Better

| Scenario | Why Direct Is Better |
|----------|---------------------|
| **Lambda-only deployment** | No extra dependencies, smaller image |
| **Simple API** | 1-3 endpoints don't need a framework |
| **Cold start optimization** | Fewer imports = faster startup |
| **Learning** | Easier to understand what's happening |

### The Trade-off We Made

```
FastAPI + Mangum:
✓ Reused existing api.py (no rewrite)
✓ Can still run locally with Docker
✓ Auto-generated docs at /docs
✗ Extra dependency (Mangum)
✗ lifespan="off" workaround for model loading
✗ More complex debugging

Direct Handler:
✓ Simpler, fewer dependencies
✓ Faster cold starts
✓ Full control over request/response
✗ Would need to rewrite api.py
✗ No local dev server (unless you mock Lambda events)
✗ No automatic docs
```

### Bottom Line

For a Lambda-only project, a direct handler (like the Egyptian project) is cleaner. We used FastAPI because the code already existed for Docker deployment and we wanted to keep both options working.


