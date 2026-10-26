import boto3
from botocore.exceptions import ClientError

REGION = "ap-southeast-2"
MODEL_ID = "amazon.nova-pro-v1:0"

client = boto3.client(
    "bedrock-runtime",
    region_name=REGION
)

try:
    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": "Reply with exactly: Nova Pro connection successful."
                    }
                ]
            }
        ],
        inferenceConfig={
            "maxTokens": 50,
            "temperature": 0
        }
    )

    print(response["output"]["message"]["content"][0]["text"])

except ClientError as error:
    print("AWS error:", error.response["Error"]["Code"])
    print("Details:", error.response["Error"]["Message"])
