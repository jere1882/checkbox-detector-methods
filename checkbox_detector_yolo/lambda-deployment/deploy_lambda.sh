#!/bin/bash

# ============================================================================
# AWS Lambda Deployment Script for Checkbox Detector
# ============================================================================
# This script builds and deploys the checkbox detector model to AWS Lambda
# 
# SAFETY: This script will verify AWS account before deploying
# ============================================================================

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================
FUNCTION_NAME="checkbox-detector"
REGION="us-east-2"
PROFILE="egyptian-project"  # AWS profile name
ECR_REPOSITORY="checkbox-detector"
IMAGE_TAG="latest"

# Lambda configuration (ML models need more resources)
MEMORY_SIZE=3008  # Max 3008 MB for Lambda
TIMEOUT=900       # 15 minutes max (900 seconds)

# ============================================================================
# SAFETY CHECKS
# ============================================================================

echo "=========================================="
echo "AWS Lambda Deployment - Safety Check"
echo "=========================================="
echo ""

# Get AWS account ID and verify
if [ -z "$PROFILE" ]; then
    echo "Using default AWS profile"
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to get AWS account. Make sure AWS CLI is configured."
        echo "Run: aws configure"
        exit 1
    fi
    USER_ARN=$(aws sts get-caller-identity --query Arn --output text 2>/dev/null)
else
    echo "Using AWS profile: $PROFILE"
    ACCOUNT_ID=$(aws sts get-caller-identity --profile ${PROFILE} --query Account --output text 2>/dev/null)
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to get AWS account with profile '$PROFILE'"
        echo "Make sure the profile exists: aws configure --profile $PROFILE"
        exit 1
    fi
    USER_ARN=$(aws sts get-caller-identity --profile ${PROFILE} --query Arn --output text 2>/dev/null)
fi

echo ""
echo "⚠️  DEPLOYMENT TARGET VERIFICATION"
echo "=========================================="
echo "AWS Account ID: $ACCOUNT_ID"
echo "User/Role: $USER_ARN"
echo "Region: $REGION"
echo "Function Name: $FUNCTION_NAME"
echo "ECR Repository: $ECR_REPOSITORY"
echo ""
echo "⚠️  WARNING: This will create/update resources in the above AWS account!"
echo ""

# Confirm before proceeding
read -p "Type 'yes' to confirm you want to deploy to this account: " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo "Proceeding with deployment..."
echo ""

# ============================================================================
# BUILD DOCKER IMAGE
# ============================================================================

echo "Step 1: Building Docker image for Lambda..."

# Get the parent directory (checkbox_detector_yolo)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

if [ ! -f "${SCRIPT_DIR}/Dockerfile.lambda" ]; then
    echo "ERROR: Dockerfile.lambda not found!"
    exit 1
fi

if [ ! -f "${PARENT_DIR}/runs/detect/train/weights/best.pt" ]; then
    echo "⚠️  WARNING: Model weights not found at ${PARENT_DIR}/runs/detect/train/weights/best.pt"
    echo "   The Lambda function will need model weights from S3 or the image will fail."
    read -p "Continue anyway? (yes/no): " CONTINUE
    if [ "$CONTINUE" != "yes" ]; then
        exit 0
    fi
fi

# Build from parent directory so we can access all files
cd "${PARENT_DIR}"
# Use docker (user is in docker group, no sudo needed)
docker build -f ${SCRIPT_DIR}/Dockerfile.lambda -t ${ECR_REPOSITORY}:${IMAGE_TAG} .

# ============================================================================
# CREATE ECR REPOSITORY
# ============================================================================

echo ""
echo "Step 2: Creating ECR repository if it doesn't exist..."

if [ -z "$PROFILE" ]; then
    aws ecr create-repository --repository-name ${ECR_REPOSITORY} --region ${REGION} 2>/dev/null || echo "Repository already exists"
else
    aws ecr create-repository --profile ${PROFILE} --repository-name ${ECR_REPOSITORY} --region ${REGION} 2>/dev/null || echo "Repository already exists"
fi

# ============================================================================
# LOGIN TO ECR
# ============================================================================

echo ""
echo "Step 3: Logging in to ECR..."

if [ -z "$PROFILE" ]; then
    aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com
else
    aws ecr get-login-password --profile ${PROFILE} --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com
fi

# ============================================================================
# TAG AND PUSH IMAGE
# ============================================================================

echo ""
echo "Step 4: Tagging and pushing image to ECR..."

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${ECR_URI}
docker push ${ECR_URI}

echo "Image pushed successfully: ${ECR_URI}"

# ============================================================================
# CREATE IAM ROLE (if needed)
# ============================================================================

echo ""
echo "Step 5: Setting up IAM role..."

ROLE_NAME="lambda-execution-role"

if [ -z "$PROFILE" ]; then
    # Check if role exists
    aws iam get-role --role-name $ROLE_NAME --region ${REGION} 2>/dev/null || {
        echo "Creating IAM role..."
        aws iam create-role \
            --role-name $ROLE_NAME \
            --assume-role-policy-document '{
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "lambda.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }'
        
        echo "Attaching execution policy..."
        aws iam attach-role-policy \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    }
else
    aws iam get-role --profile ${PROFILE} --role-name $ROLE_NAME --region ${REGION} 2>/dev/null || {
        echo "Creating IAM role..."
        aws iam create-role \
            --profile ${PROFILE} \
            --role-name $ROLE_NAME \
            --assume-role-policy-document '{
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "lambda.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }'
        
        echo "Attaching execution policy..."
        aws iam attach-role-policy \
            --profile ${PROFILE} \
            --role-name $ROLE_NAME \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    }
fi

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# ============================================================================
# CREATE OR UPDATE LAMBDA FUNCTION
# ============================================================================

echo ""
echo "Step 6: Creating or updating Lambda function..."

if [ -z "$PROFILE" ]; then
    # Try to create function, if it exists, update the code
    aws lambda create-function \
        --function-name ${FUNCTION_NAME} \
        --package-type Image \
        --code ImageUri=${ECR_URI} \
        --timeout ${TIMEOUT} \
        --memory-size ${MEMORY_SIZE} \
        --role ${ROLE_ARN} \
        --region ${REGION} 2>/dev/null || {
        echo "Function exists, updating code..."
        aws lambda update-function-code \
            --function-name ${FUNCTION_NAME} \
            --image-uri ${ECR_URI} \
            --region ${REGION}
        
        echo "Updating function configuration..."
        aws lambda update-function-configuration \
            --function-name ${FUNCTION_NAME} \
            --timeout ${TIMEOUT} \
            --memory-size ${MEMORY_SIZE} \
            --region ${REGION}
    }
else
    aws lambda create-function \
        --profile ${PROFILE} \
        --function-name ${FUNCTION_NAME} \
        --package-type Image \
        --code ImageUri=${ECR_URI} \
        --timeout ${TIMEOUT} \
        --memory-size ${MEMORY_SIZE} \
        --role ${ROLE_ARN} \
        --region ${REGION} 2>/dev/null || {
        echo "Function exists, updating code..."
        aws lambda update-function-code \
            --profile ${PROFILE} \
            --function-name ${FUNCTION_NAME} \
            --image-uri ${ECR_URI} \
            --region ${REGION}
        
        echo "Updating function configuration..."
        aws lambda update-function-configuration \
            --profile ${PROFILE} \
            --function-name ${FUNCTION_NAME} \
            --timeout ${TIMEOUT} \
            --memory-size ${MEMORY_SIZE} \
            --region ${REGION}
    }
fi

# ============================================================================
# WAIT FOR FUNCTION TO BE READY
# ============================================================================

echo ""
echo "Step 7: Waiting for function to be ready..."

if [ -z "$PROFILE" ]; then
    aws lambda wait function-updated --function-name ${FUNCTION_NAME} --region ${REGION}
    FUNCTION_ARN=$(aws lambda get-function --function-name ${FUNCTION_NAME} --region ${REGION} --query 'Configuration.FunctionArn' --output text)
else
    aws lambda wait function-updated --profile ${PROFILE} --function-name ${FUNCTION_NAME} --region ${REGION}
    FUNCTION_ARN=$(aws lambda get-function --profile ${PROFILE} --function-name ${FUNCTION_NAME} --region ${REGION} --query 'Configuration.FunctionArn' --output text)
fi

# ============================================================================
# DEPLOYMENT COMPLETE
# ============================================================================

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Function Name: ${FUNCTION_NAME}"
echo "Function ARN: ${FUNCTION_ARN}"
echo "Region: ${REGION}"
echo ""
echo "Next steps:"
echo "1. Create API Gateway to expose the function (see setup_api_gateway.sh)"
echo "2. Test the function:"
if [ -z "$PROFILE" ]; then
    echo "   aws lambda invoke --function-name ${FUNCTION_NAME} --region ${REGION} response.json"
else
    echo "   aws lambda invoke --profile ${PROFILE} --function-name ${FUNCTION_NAME} --region ${REGION} response.json"
fi
echo ""
echo "Note: If model weights are not in the image, upload to S3 and modify"
echo "      lambda_handler.py to download them at startup."
echo ""

