import boto3
import json

# Create a SageMaker runtime client
runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Replace this with your actual input
input_data = {
    "feature1": 0.5,
    "feature2": 1.2,
    "feature3": 3.0,
    # Add more based on your model input
}

# Invoke the endpoint
response = runtime.invoke_endpoint(
    EndpointName='evidence-predictor-endpoint',  # your endpoint name
    ContentType='application/json',
    Body=json.dumps(input_data)
)

# Decode the response
result = response['Body'].read().decode('utf-8')
print("Prediction result:", result)
