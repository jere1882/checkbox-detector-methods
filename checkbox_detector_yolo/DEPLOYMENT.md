# Deployment Guide: FastAPI + Docker

This guide explains how the checkbox detection model is deployed using FastAPI and Docker, covering the concepts and architecture from first principles.

## Overview: The Full Stack

Your checkbox detection model goes through several layers to become a production service:

1. **Python Model** → Your YOLO model that detects checkboxes
2. **FastAPI** → Wraps the model with HTTP endpoints
3. **Docker** → Packages everything into a container
4. **Docker Compose** → Orchestrates the deployment

## FastAPI: Wrapping Your Model with HTTP

FastAPI is a Python web framework that turns your model into an HTTP API. Instead of running your model as a Python script, FastAPI lets you invoke it over the network using standard HTTP requests.

### What FastAPI Does

In `api.py`, you define:
- **Endpoints**: URL paths that accept requests (like `/predict`, `/health`)
- **Request handling**: How to process incoming data (images, parameters)
- **Response formatting**: How to return results (JSON, images)

When you run `python api.py`, FastAPI starts a web server that:
- Listens on a port (8000)
- Accepts HTTP requests
- Routes them to your Python functions
- Returns responses

### Example Flow

```
Client sends: POST http://localhost:8000/predict (with image file)
    ↓
FastAPI receives request
    ↓
Routes to your predict() function
    ↓
Your function loads image, runs model.predict()
    ↓
Returns JSON with detection results
    ↓
Client receives: {"detections": [...], "counts": {...}}
```

### Uvicorn: The ASGI Server

**Uvicorn** is the web server that actually runs your FastAPI application. Here's what it does:

- **ASGI (Asynchronous Server Gateway Interface)**: The protocol that lets Python web frameworks communicate with web servers
- **Handles HTTP**: Listens for incoming HTTP requests, parses them, and passes them to FastAPI
- **Async support**: Can handle multiple requests concurrently (important for production)
- **Process management**: Manages worker processes, handles crashes, etc.

In your code:
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

This tells uvicorn to:
- Run your FastAPI `app`
- Listen on all network interfaces (`0.0.0.0`)
- Use port 8000

**Why not just run FastAPI directly?** FastAPI is the framework (defines routes, handles logic), but uvicorn is the server (handles network, processes requests). You need both:
- FastAPI = The application logic
- Uvicorn = The server that runs it

In production, you might use multiple uvicorn workers or a reverse proxy (nginx) in front of it for better performance and reliability.

## Docker: Packaging Everything

### Docker Image vs Container: The Key Distinction

**Docker Image** = A blueprint/template
- A read-only snapshot containing:
  - Operating system (or parts of it)
  - Python and its version
  - All dependencies (PyTorch, OpenCV, FastAPI, etc.)
  - Your code
  - Configuration files
- Stored on disk
- Immutable (can't change it once built)
- Can be shared, versioned, pushed to registries

**Docker Container** = A running instance
- Created from an image
- Has its own isolated filesystem
- Runs as a process on your machine
- Can be started, stopped, deleted
- Ephemeral (changes are lost when container stops, unless in volumes)

**Analogy**: 
- Image = A recipe/blueprint for a house
- Container = An actual house built from that blueprint (you can have multiple houses from one blueprint)

### The Dockerfile: Building Instructions

The `Dockerfile` is a recipe that tells Docker how to build an image:

```dockerfile
FROM python:3.9-slim          # Start with base image (Python 3.9 on Linux)
WORKDIR /app                   # Set working directory inside container
RUN apt-get update...          # Install system libraries (OpenCV needs these)
COPY requirements.txt .        # Copy dependency list
RUN pip install -r requirements.txt  # Install Python packages
COPY . .                       # Copy your code
EXPOSE 8000                    # Document that app uses port 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]  # Run command
```

When you run `docker build`, Docker:
1. Starts with the base image (`python:3.9-slim`)
2. Executes each instruction sequentially
3. Creates intermediate layers (for caching)
4. Produces a final image

### docker-compose.yml: Orchestration

`docker-compose.yml` configures how to run containers:

```yaml
services:
  checkbox-detector-api:
    build: .                    # Build from Dockerfile
    ports:
      - "8000:8000"            # Map host:container ports
    volumes:
      - ./runs/...:/app/runs/...  # Mount local files into container
    environment:
      - PYTHONUNBUFFERED=1     # Set environment variables
```

This tells Docker Compose:
- How to build the image
- Which ports to expose
- Which files to mount (so container can access your model weights)
- Environment variables to set

## The Kitchen Analogy

Think of deploying your model like running a restaurant:

- **Your Model** = The chef (does the actual work - detects checkboxes)
- **FastAPI** = The waiter (takes orders via HTTP, brings back results)
- **Uvicorn** = The restaurant manager (handles seating, manages staff, coordinates service)
- **Docker Image** = The kitchen blueprint (specifies all equipment, ingredients, layout)
- **Docker Container** = The actual kitchen (running instance with all tools ready)
- **Dockerfile** = The construction plans (how to build the kitchen)
- **docker-compose.yml** = The restaurant setup (tables, connections, reservations)

When you run `docker-compose up`:
1. **Builds the kitchen** (creates image from Dockerfile)
2. **Opens the restaurant** (starts container)
3. **Sets up tables** (maps ports, mounts volumes)
4. **Manager starts** (uvicorn runs)
5. **Waiter is ready** (FastAPI endpoints active)
6. **Customers can order** (HTTP requests come in)
7. **Chef prepares food** (model processes images)
8. **Waiter delivers** (results returned)

## What `docker-compose up` Actually Does

When you run `docker-compose up`, here's the step-by-step process:

1. **Checks for image**: "Does the image exist?" 
   - If not → runs `docker build` (reads Dockerfile, creates image)
   - If yes → uses existing image

2. **Creates container**: Spins up a new container instance from the image
   - Container gets its own isolated filesystem
   - Container gets its own network namespace
   - Container runs as a process on your machine

3. **Mounts volumes**: Connects your local files to container
   - `./runs/detect/train/weights` → `/app/runs/detect/train/weights`
   - This lets the container access your model weights without copying them

4. **Maps ports**: Connects network ports
   - Host port 8000 → Container port 8000
   - When you access `localhost:8000`, it routes to the container

5. **Runs CMD**: Executes the command from Dockerfile
   - `uvicorn api:app --host 0.0.0.0 --port 8000`
   - This starts the FastAPI server inside the container

6. **FastAPI starts**: Server begins listening
   - Uvicorn starts the ASGI server
   - FastAPI app loads
   - Model loads on startup (from `@app.on_event("startup")`)
   - Server is ready to accept requests

## Why Use Docker?

### Reproducibility
- Same environment everywhere (dev, staging, production)
- No "works on my machine" problems
- Dependencies are frozen

### Isolation
- Doesn't pollute your system Python
- Can run different Python versions side-by-side
- Easy to clean up (just delete container)

### Deployment
- Push image to registry (Docker Hub, AWS ECR, etc.)
- Pull and run on any server
- No need to install dependencies on server

### Scalability
- Easy to run multiple containers
- Can use Kubernetes, Docker Swarm for orchestration
- Load balancing across containers

## Common Commands

```bash
# Build the image
docker build -t checkbox-detector:latest .

# Run a container (manually)
docker run -p 8000:8000 checkbox-detector:latest

# Use docker-compose (recommended)
docker-compose up -d          # Start in background
docker-compose up              # Start in foreground (see logs)
docker-compose down            # Stop and remove containers
docker-compose logs -f         # View logs
docker-compose ps              # List running containers

# Inspect what's running
docker ps                      # List all running containers
docker exec checkbox-detector-api ps aux  # See processes inside container
docker exec checkbox-detector-api ls -la /app  # See filesystem
```

## Production Considerations

### Environment Variables
Use environment variables for configuration:
```yaml
environment:
  - MODEL_PATH=/app/runs/detect/train/weights/best.pt
  - CONF_THRESHOLD=0.2
```

### Health Checks
Your Dockerfile includes a health check:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"
```

This lets Docker (and orchestrators like Kubernetes) know if your service is healthy.

### Resource Limits
In production, set resource limits:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

### Multiple Workers
For better performance, run multiple uvicorn workers:
```dockerfile
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Reverse Proxy
In production, use nginx or similar in front of uvicorn:
- Handles SSL/TLS
- Load balancing
- Static file serving
- Better security

## Troubleshooting

### Container won't start
```bash
docker-compose logs              # Check logs
docker-compose up --build         # Rebuild image
```

### Port already in use
Change port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Use 8001 on host instead
```

### Model not found
Check volume mounts:
```bash
docker exec checkbox-detector-api ls -la /app/runs/detect/train/weights
```

### Can't access API
Check if container is running:
```bash
docker ps
curl http://localhost:8000/health
```

## Port Forwarding and Network Isolation

### Understanding Port Mapping

When you run FastAPI directly (without Docker), it's straightforward:
```
Your Machine:
  └─ Python process listens on localhost:8000
  └─ You access: http://localhost:8000
```

With Docker, there's an additional layer of network isolation:

```
Your Machine (Host):
  └─ Port 8000 (host port)
      ↓ (port mapping/forwarding)
  └─ Docker Container:
      └─ Port 8000 (container port)
      └─ FastAPI listening on 0.0.0.0:8000 inside container
```

In your `docker-compose.yml`:
```yaml
ports:
  - "8000:8000"  # host:container
```

This means:
- **Left side (`8000`)** = port on your host machine
- **Right side (`8000`)** = port inside the container
- Docker creates a network bridge that forwards traffic between them

### The Network Isolation Layer

Containers have their own **network namespace**. When FastAPI listens on `0.0.0.0:8000` inside the container, it's only accessible within that container's network by default. Port mapping exposes it to the host.

**Request Flow:**
1. You send request to `localhost:8000` (on your host)
2. Docker's network bridge intercepts it
3. Forwards to container's port 8000
4. FastAPI receives it inside the container
5. Response goes back through the same path

**Why `0.0.0.0` in the Dockerfile?**
- `0.0.0.0` means "listen on all network interfaces"
- If you used `127.0.0.1` (localhost only), it would only listen inside the container and wouldn't accept forwarded traffic
- `0.0.0.0` allows the container to accept connections from outside (via port mapping)

**You could use different ports:**
```yaml
ports:
  - "9000:8000"  # Access via localhost:9000, forwards to container:8000
```

### Without Docker vs With Docker

**Without Docker:**
- Process runs directly on your machine
- Listens on `localhost:8000`
- Direct access, no indirection

**With Docker:**
- Process runs inside isolated container
- Listens on `0.0.0.0:8000` inside container
- Port mapping adds one level of indirection (host port → container port)
- This isolation is the benefit: container can't affect your host network

## Deployment Scenarios

Once you have a containerized model, you can deploy it in various scenarios. The same Docker image can run in different environments with different orchestration.

### Scenario A: Robot/Edge Deployment

For a robot or edge device, you'd typically deploy directly on the robot's computer.

**Option 1: Direct Docker on Robot**
```bash
# On robot's computer
docker load < checkbox-detector.tar  # Load pre-built image
docker run -d --name detector \
  --network host \  # Use host network (simpler for local services)
  -v /robot/models:/app/runs/detect/train/weights:ro \
  checkbox-detector:latest
```

**Option 2: No Docker (Direct Python)**
- Install Python + dependencies on robot
- Run `python api.py` directly
- Robot's other services call `http://localhost:8000`

**Option 3: ROS Integration (if using ROS)**
- Wrap FastAPI in a ROS node
- Publish detections as ROS messages
- Other nodes subscribe to checkbox detections

**Considerations for robots:**
- Resource constraints (CPU, RAM, GPU)
- Network: might use `--network host` for simplicity
- Updates: push new images, restart container
- Reliability: auto-restart policies, health monitoring

**Key Point:** Whether it's Docker or just FastAPI+Python, you have a **process listening on a port**. Docker adds isolation, but the end result is the same: a service that other processes can call.

### Scenario B: SaaS/Cloud Deployment

For cloud deployment, you have several options depending on your needs.

#### Option 1: Container Registry + Cloud Services

**Step 1: Push to Registry**
```bash
# Build and tag
docker build -t checkbox-detector:latest .
docker tag checkbox-detector:latest gcr.io/my-project/checkbox-detector:v1

# Push to registry
docker push gcr.io/my-project/checkbox-detector:v1
```

**Step 2a: Google Cloud Run (Serverless Containers)**
```bash
gcloud run deploy checkbox-detector \
  --image gcr.io/my-project/checkbox-detector:v1 \
  --platform managed \
  --region us-central1 \
  --port 8000 \
  --memory 2Gi \
  --cpu 2
```
- Auto-scales from 0 to N instances
- Pay per request
- HTTPS endpoint automatically
- No infrastructure management

**Step 2b: AWS ECS (Elastic Container Service)**
```yaml
# task-definition.json
{
  "family": "checkbox-detector",
  "containerDefinitions": [{
    "name": "api",
    "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/checkbox-detector:latest",
    "portMappings": [{"containerPort": 8000}],
    "memory": 2048,
    "cpu": 1024
  }]
}
```
- Deploy to ECS cluster
- Behind Application Load Balancer
- Auto-scaling groups

**Step 2c: Kubernetes**
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: checkbox-detector
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: gcr.io/my-project/checkbox-detector:v1
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: checkbox-detector-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: checkbox-detector
```
- Deploy to GKE, EKS, AKS
- Load balancing, auto-scaling, rolling updates

#### Option 2: VM Deployment

```bash
# On cloud VM (AWS EC2, GCP Compute Engine, Azure VM)
# Install Docker
sudo apt-get update
sudo apt-get install docker.io

# Pull and run
docker pull your-registry/checkbox-detector:latest
docker run -d -p 80:8000 \
  --restart unless-stopped \
  your-registry/checkbox-detector:latest

# Set up nginx reverse proxy for HTTPS
# Configure firewall rules
```

#### Option 3: Serverless Functions (if latency acceptable)

For very lightweight use, could wrap in AWS Lambda / Cloud Functions:
- Package model + code
- API Gateway triggers function
- Higher latency (cold starts), but very cheap

### Cloud Run vs AWS Lambda

**Google Cloud Run** and **AWS Lambda** are similar but different:

| Feature | Google Cloud Run | AWS Lambda |
|---------|-----------------|------------|
| **Container Support** | ✅ Full Docker containers | ⚠️ Limited (zip packages, or Lambda containers) |
| **Cold Starts** | ~1-2 seconds | ~100ms-5s (varies) |
| **Max Execution Time** | 60 minutes (configurable) | 15 minutes max |
| **Memory** | Up to 8GB | Up to 10GB |
| **Concurrency** | Multiple requests per instance | 1 request per instance (by default) |
| **Use Case** | Containerized apps, APIs | Event-driven functions |
| **Deployment** | `docker push` then deploy | Upload zip or container image |

**For your checkbox detector:**
- **Cloud Run**: Better fit. You can deploy your Docker container as-is, longer timeouts, handles concurrent requests.
- **Lambda**: Possible but less ideal. Cold starts can be slow with ML models, 15-minute limit, and you'd need to package the model differently.

**Analogy:**
- Cloud Run = "Run my container, scale it automatically"
- Lambda = "Run my function, scale it automatically"

### Load Balancers and Auto-Scaling

#### AWS Lambda
**No load balancer needed.**
- API Gateway or Function URL provides the HTTPS endpoint
- AWS handles routing and scaling automatically
- Each request can trigger a new instance if needed

```
Internet → API Gateway → Lambda (auto-scales)
```

#### AWS ECS
**Usually needs a load balancer (ALB).**

**Option 1: ECS Fargate with Application Load Balancer (ALB)**
```
Internet → ALB → ECS Tasks (auto-scales based on CPU/memory)
```
- ALB distributes traffic across tasks
- ECS auto-scales tasks (adds/removes containers)
- You configure auto-scaling rules

**Option 2: ECS with Service Discovery**
```
Internet → Route 53 / Service Discovery → ECS Tasks
```
- For internal services, you might use service discovery instead of ALB

**Why ALB with ECS?**
- Distributes traffic across tasks
- Provides health checks
- Handles SSL/TLS termination
- Provides routing rules

Without ALB, you'd need to manually point to specific task IPs (not scalable).

#### Google Cloud Run
**No load balancer needed.**
- Built-in load balancing
- Auto-scales from 0 to N instances
- HTTPS endpoint automatically
- Handles load balancing internally

### SaaS Architecture Diagram

**With Lambda:**
```
Internet
  ↓
API Gateway (HTTPS, routing, auth)
  ↓
Lambda Functions (auto-scales 0 to thousands)
  └─ Your checkbox detector code
```

**With ECS:**
```
Internet
  ↓
Application Load Balancer (HTTPS, routing)
  ↓
ECS Service (auto-scales tasks based on metrics)
  ├─ Task 1: Container
  ├─ Task 2: Container
  └─ Task 3: Container
  (adds/removes tasks automatically)
```

**With Cloud Run:**
```
Internet
  ↓
Cloud Run Service (HTTPS, auto-scales 0 to N)
  └─ Your container (instances created/destroyed automatically)
```

### Auto-Scaling Details

**Lambda:**
- Scales automatically per request
- No configuration needed
- Can scale to thousands concurrently

**ECS:**
- You configure auto-scaling policies:
  ```json
  {
    "minCapacity": 1,
    "maxCapacity": 10,
    "targetTrackingScalingPolicies": [{
      "targetValue": 70.0,
      "predefinedMetricSpecification": {
        "predefinedMetricType": "ECSServiceAverageCPUUtilization"
      }
    }]
  }
  ```
- ALB distributes traffic across running tasks
- ECS adds/removes tasks based on CPU, memory, or custom metrics

**Cloud Run:**
- Auto-scales based on requests
- Can scale to 0 (no cost when idle)
- Configurable min/max instances
- Handles load balancing internally

### Do You Need a Load Balancer?

| Service | Load Balancer Needed? | Why |
|---------|----------------------|-----|
| **Lambda** | No | API Gateway handles it |
| **Cloud Run** | No | Built-in |
| **ECS** | Usually Yes (ALB) | For public-facing services, ALB distributes traffic, handles SSL, provides health checks |

### Key Differences: Robot vs SaaS

| Aspect | Robot | SaaS/Cloud |
|--------|-------|------------|
| **Location** | On robot hardware | Cloud servers |
| **Network** | Local (localhost or LAN) | Internet (HTTPS) |
| **Scaling** | Single instance | Multiple instances, auto-scale |
| **Updates** | Manual push/SSH | CI/CD pipeline, rolling updates |
| **Monitoring** | Local logs | Cloud monitoring (CloudWatch, Stackdriver) |
| **Cost** | One-time hardware | Pay per usage/instance |
| **Latency** | Very low (<1ms) | Network latency (50-200ms) |

### Production Considerations for Both

**Security:**
- Authentication/API keys
- Rate limiting
- Input validation
- HTTPS/TLS

**Reliability:**
- Health checks
- Auto-restart on failure
- Graceful shutdown
- Circuit breakers

**Monitoring:**
- Logs (centralized logging)
- Metrics (request rate, latency, errors)
- Alerts (downtime, high error rate)

## Summary

- **FastAPI**: Wraps your model with HTTP endpoints
- **Uvicorn**: The ASGI server that runs FastAPI
- **Docker Image**: Blueprint/template (read-only, stored on disk)
- **Docker Container**: Running instance (ephemeral, isolated environment)
- **Dockerfile**: Instructions to build an image
- **docker-compose.yml**: Configuration to run containers
- **docker-compose up**: Builds image (if needed), creates container, runs your API
- **Port Mapping**: Forwards host ports to container ports (network isolation layer)
- **Deployment**: Same container can run on robots (edge) or in cloud (SaaS) with different orchestration

The result: Your model is accessible via HTTP, packaged in a reproducible, isolated environment that can run anywhere Docker is installed—from a robot to the cloud.

