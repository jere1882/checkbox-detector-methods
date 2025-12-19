#!/bin/bash

# ============================================================================
# API Gateway Setup Script for Lambda Function
# ============================================================================
# This script creates an API Gateway REST API connected to the Lambda function
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION - MUST MATCH deploy_lambda.sh
# ============================================================================
FUNCTION_NAME="checkbox-detector"
REGION="us-east-2"
PROFILE="egyptian-project"  # AWS profile name
API_NAME="${FUNCTION_NAME}-API"

# ============================================================================
# GET AWS ACCOUNT INFO
# ============================================================================

if [ -z "$PROFILE" ]; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
else
    ACCOUNT_ID=$(aws sts get-caller-identity --profile ${PROFILE} --query Account --output text)
fi

echo "=========================================="
echo "API Gateway Setup"
echo "=========================================="
echo "Account ID: $ACCOUNT_ID"
echo "Region: $REGION"
echo "Function: $FUNCTION_NAME"
echo ""

# ============================================================================
# CREATE API GATEWAY
# ============================================================================

echo "Step 1: Creating API Gateway REST API..."

if [ -z "$PROFILE" ]; then
    API_ID=$(aws apigateway create-rest-api \
        --name ${API_NAME} \
        --description "API for Checkbox Detector Lambda Function" \
        --region ${REGION} \
        --query 'id' \
        --output text)
else
    API_ID=$(aws apigateway create-rest-api \
        --profile ${PROFILE} \
        --name ${API_NAME} \
        --description "API for Checkbox Detector Lambda Function" \
        --region ${REGION} \
        --query 'id' \
        --output text)
fi

echo "API Gateway ID: $API_ID"

# ============================================================================
# GET ROOT RESOURCE
# ============================================================================

echo ""
echo "Step 2: Getting root resource..."

if [ -z "$PROFILE" ]; then
    ROOT_RESOURCE_ID=$(aws apigateway get-resources \
        --rest-api-id ${API_ID} \
        --region ${REGION} \
        --query 'items[?path==`/`].id' \
        --output text)
else
    ROOT_RESOURCE_ID=$(aws apigateway get-resources \
        --profile ${PROFILE} \
        --rest-api-id ${API_ID} \
        --region ${REGION} \
        --query 'items[?path==`/`].id' \
        --output text)
fi

# ============================================================================
# CREATE PROXY RESOURCE
# ============================================================================

echo ""
echo "Step 3: Creating proxy resource..."

if [ -z "$PROFILE" ]; then
    PROXY_RESOURCE_ID=$(aws apigateway create-resource \
        --rest-api-id ${API_ID} \
        --parent-id ${ROOT_RESOURCE_ID} \
        --path-part '{proxy+}' \
        --region ${REGION} \
        --query 'id' \
        --output text)
else
    PROXY_RESOURCE_ID=$(aws apigateway create-resource \
        --profile ${PROFILE} \
        --rest-api-id ${API_ID} \
        --parent-id ${ROOT_RESOURCE_ID} \
        --path-part '{proxy+}' \
        --region ${REGION} \
        --query 'id' \
        --output text)
fi

# ============================================================================
# CREATE ANY METHOD
# ============================================================================

echo ""
echo "Step 4: Creating ANY method for proxy..."

LAMBDA_URI="arn:aws:apigateway:${REGION}:lambda:path/2015-03-31/functions/arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}/invocations"

if [ -z "$PROFILE" ]; then
    aws apigateway put-method \
        --rest-api-id ${API_ID} \
        --resource-id ${PROXY_RESOURCE_ID} \
        --http-method ANY \
        --authorization-type NONE \
        --region ${REGION}
    
    aws apigateway put-integration \
        --rest-api-id ${API_ID} \
        --resource-id ${PROXY_RESOURCE_ID} \
        --http-method ANY \
        --type AWS_PROXY \
        --integration-http-method POST \
        --uri ${LAMBDA_URI} \
        --region ${REGION}
    
    # Also create method on root
    aws apigateway put-method \
        --rest-api-id ${API_ID} \
        --resource-id ${ROOT_RESOURCE_ID} \
        --http-method ANY \
        --authorization-type NONE \
        --region ${REGION}
    
    aws apigateway put-integration \
        --rest-api-id ${API_ID} \
        --resource-id ${ROOT_RESOURCE_ID} \
        --http-method ANY \
        --type AWS_PROXY \
        --integration-http-method POST \
        --uri ${LAMBDA_URI} \
        --region ${REGION}
else
    aws apigateway put-method \
        --profile ${PROFILE} \
        --rest-api-id ${API_ID} \
        --resource-id ${PROXY_RESOURCE_ID} \
        --http-method ANY \
        --authorization-type NONE \
        --region ${REGION}
    
    aws apigateway put-integration \
        --profile ${PROFILE} \
        --rest-api-id ${API_ID} \
        --resource-id ${PROXY_RESOURCE_ID} \
        --http-method ANY \
        --type AWS_PROXY \
        --integration-http-method POST \
        --uri ${LAMBDA_URI} \
        --region ${REGION}
    
    # Also create method on root
    aws apigateway put-method \
        --profile ${PROFILE} \
        --rest-api-id ${API_ID} \
        --resource-id ${ROOT_RESOURCE_ID} \
        --http-method ANY \
        --authorization-type NONE \
        --region ${REGION}
    
    aws apigateway put-integration \
        --profile ${PROFILE} \
        --rest-api-id ${API_ID} \
        --resource-id ${ROOT_RESOURCE_ID} \
        --http-method ANY \
        --type AWS_PROXY \
        --integration-http-method POST \
        --uri ${LAMBDA_URI} \
        --region ${REGION}
fi

# ============================================================================
# ADD LAMBDA PERMISSION
# ============================================================================

echo ""
echo "Step 5: Adding Lambda permission for API Gateway..."

if [ -z "$PROFILE" ]; then
    aws lambda add-permission \
        --function-name ${FUNCTION_NAME} \
        --statement-id apigateway-invoke \
        --action lambda:InvokeFunction \
        --principal apigateway.amazonaws.com \
        --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*/*" \
        --region ${REGION} 2>/dev/null || echo "Permission already exists"
else
    aws lambda add-permission \
        --profile ${PROFILE} \
        --function-name ${FUNCTION_NAME} \
        --statement-id apigateway-invoke \
        --action lambda:InvokeFunction \
        --principal apigateway.amazonaws.com \
        --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*/*" \
        --region ${REGION} 2>/dev/null || echo "Permission already exists"
fi

# ============================================================================
# DEPLOY API
# ============================================================================

echo ""
echo "Step 6: Deploying API to 'prod' stage..."

if [ -z "$PROFILE" ]; then
    aws apigateway create-deployment \
        --rest-api-id ${API_ID} \
        --stage-name prod \
        --region ${REGION}
else
    aws apigateway create-deployment \
        --profile ${PROFILE} \
        --rest-api-id ${API_ID} \
        --stage-name prod \
        --region ${REGION}
fi

# ============================================================================
# GET API URL
# ============================================================================

API_URL="https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod"

echo ""
echo "=========================================="
echo "✅ API Gateway Setup Complete!"
echo "=========================================="
echo ""
echo "API URL: ${API_URL}"
echo ""
echo "Test endpoints:"
echo "  Health: ${API_URL}/health"
echo "  Predict: ${API_URL}/predict (POST with image)"
echo ""
echo "Example curl command:"
echo "  curl -X POST ${API_URL}/health"
echo ""

