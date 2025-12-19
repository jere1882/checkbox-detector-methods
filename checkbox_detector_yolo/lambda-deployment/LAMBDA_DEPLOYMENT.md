# AWS Lambda Deployment Guide

This guide explains how to deploy the checkbox detector model to AWS Lambda using Docker containers.

## Prerequisites

### AWS Credentials

You need AWS credentials configured. The deployment script will verify the account before deploying.

**Option 1: Default Profile**
```bash
aws configure
# Enter your Access Key ID, Secret Access Key, region, and output format
```

**Option 2: Named Profile**
```bash
aws configure --profile your-profile-name
```

**Required Permissions:**
- ECR: Create repository, push images
- Lambda: Create/update functions
- IAM: Create roles and attach policies
- API Gateway: Create APIs (if using API Gateway setup script)

### Docker

Docker must be installed and running:
```bash
sudo docker --version
```

### Model Weights

The model weights should be at:
```
runs/detect/train/weights/best.pt
```

If not present, the script will warn you. You'll need to either:
1. Include weights in the Docker image (recommended for Lambda)
2. Upload to S3 and modify `lambda_handler.py` to download at startup

## Deployment Steps

### Step 1: Review Configuration

Edit `lambda-deployment/deploy_lambda.sh` and update these variables:

```bash
FUNCTION_NAME="checkbox-detector"  # Lambda function name
REGION="us-east-2"                 # AWS region
PROFILE=""                          # AWS profile (empty for default)
ECR_REPOSITORY="checkbox-detector" # ECR repository name
MEMORY_SIZE=3008                    # Lambda memory (max 3008 MB)
TIMEOUT=900                         # Timeout in seconds (max 900 = 15 min)
```

**Important:** The script will show you the AWS account ID and ask for confirmation before deploying.

### Step 2: Deploy Lambda Function

```bash
cd lambda-deployment
./deploy_lambda.sh
```

Or from the parent directory:
```bash
./lambda-deployment/deploy_lambda.sh
```

**What the script does:**
1. **Safety Check**: Shows AWS account ID and asks for confirmation
2. **Builds Docker Image**: Creates Lambda-compatible image using `Dockerfile.lambda`
3. **Creates ECR Repository**: Sets up container registry if needed
4. **Pushes Image**: Uploads image to ECR
5. **Creates IAM Role**: Sets up execution role for Lambda
6. **Creates/Updates Lambda**: Deploys or updates the function

**Safety Features:**
- Verifies AWS account before proceeding
- Requires explicit "yes" confirmation
- Shows account ID, user/role, and region
- Checks for model weights and warns if missing

### Step 3: Set Up API Gateway (Optional)

To expose the Lambda function via HTTP:

```bash
cd lambda-deployment
./setup_api_gateway.sh
```

Or from the parent directory:
```bash
./lambda-deployment/setup_api_gateway.sh
```

This creates a REST API that proxies all requests to your Lambda function.

**Note:** Make sure the `FUNCTION_NAME` and `REGION` in `setup_api_gateway.sh` match `deploy_lambda.sh`.

## Lambda-Specific Considerations

### Memory and Timeout

ML models need significant resources:
- **Memory**: 3008 MB (maximum for Lambda)
- **Timeout**: 900 seconds (15 minutes, maximum for Lambda)

These are set in `deploy_lambda.sh`. Adjust if needed, but note the limits.

### Cold Starts

Lambda has cold start latency:
- First request: ~5-10 seconds (loading model)
- Subsequent requests: ~1-2 seconds (if container is warm)

For production, consider:
- Provisioned concurrency (keeps containers warm)
- Cloud Run (better for containerized ML models)

### Model Weights

**Option 1: Include in Image (Current Approach)**
- Model weights are copied into the Docker image
- Larger image size (~100-200 MB)
- Faster cold starts (no download needed)

**Option 2: Download from S3**
If model weights are too large or you want to update them without redeploying:

1. Upload weights to S3:
```bash
aws s3 cp runs/detect/train/weights/best.pt s3://your-bucket/models/best.pt
```

2. Modify `lambda_handler.py` to download at startup:
```python
import boto3
import os

def download_model_from_s3():
    s3 = boto3.client('s3')
    s3.download_file('your-bucket', 'models/best.pt', '/tmp/best.pt')
    return '/tmp/best.pt'
```

3. Update `api.py` to use `/tmp` (Lambda's writable directory)

## Testing

### Test Lambda Function Directly

```bash
# Test with a simple event
aws lambda invoke \
    --function-name checkbox-detector \
    --payload '{"httpMethod": "GET", "path": "/health"}' \
    response.json

cat response.json
```

### Test via API Gateway

If you set up API Gateway:

```bash
# Get the API URL from setup_api_gateway.sh output
API_URL="https://YOUR_API_ID.execute-api.us-east-2.amazonaws.com/prod"

# Test health endpoint
curl ${API_URL}/health

# Test prediction (with image file)
curl -X POST ${API_URL}/predict \
    -F "file=@test_image.jpg" \
    -F "conf=0.2"
```

## Cost Considerations

**Lambda Pricing:**
- $0.0000166667 per GB-second
- $0.20 per 1M requests
- Free tier: 1M requests/month, 400,000 GB-seconds/month

**Example:**
- 1000 requests/day
- 3GB memory, 2 seconds average
- Cost: ~$0.10/month (after free tier)

**Compare to Cloud Run:**
- Better for containerized ML models
- No 15-minute timeout limit
- Better cold start handling
- Similar pricing model

## Troubleshooting

### "Model weights not found"

Either:
1. Ensure `runs/detect/train/weights/best.pt` exists before building
2. Or modify code to download from S3

### "Function timeout"

Increase timeout in `deploy_lambda.sh` (max 900 seconds). If you need longer, consider Cloud Run or ECS.

### "Out of memory"

Increase `MEMORY_SIZE` in `deploy_lambda.sh` (max 3008 MB). If still insufficient, consider Cloud Run or ECS.

### "Permission denied"

Check AWS credentials:
```bash
aws sts get-caller-identity
```

Verify IAM permissions for:
- ECR (create repository, push images)
- Lambda (create/update functions)
- IAM (create roles)

### View Logs

```bash
aws logs tail /aws/lambda/checkbox-detector --follow
```

## Comparison: Lambda vs Cloud Run

| Feature | Lambda | Cloud Run |
|---------|--------|-----------|
| **Max Memory** | 10 GB | 8 GB |
| **Max Timeout** | 15 min | 60 min |
| **Cold Starts** | 5-10s | 1-2s |
| **Container Support** | ✅ (with limitations) | ✅ Full |
| **Best For** | Event-driven, short tasks | APIs, long-running |
| **Cost** | Pay per request | Pay per request |

**Recommendation:** For ML models, Cloud Run is often better due to longer timeouts and better cold start handling. Lambda is fine for testing or if you need event-driven triggers.

## Security Notes

- The deployment script shows your AWS account ID before deploying
- Always verify you're deploying to the correct account
- Use IAM roles with least privilege
- Consider adding API keys or authentication to API Gateway
- Enable CloudWatch logging for monitoring

## Next Steps

1. **Monitor**: Set up CloudWatch alarms for errors/timeouts
2. **Optimize**: Consider provisioned concurrency for production
3. **Scale**: If hitting limits, consider Cloud Run or ECS
4. **Secure**: Add authentication/authorization to API Gateway

