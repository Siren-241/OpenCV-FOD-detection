import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REGION = "ap-southeast-2"
MODEL_ID = "amazon.nova-pro-v1:0"

BASE_DIR = Path(__file__).parent
CANDIDATE_DIR = BASE_DIR / "results" / "candidates"

BASELINE_PATH = CANDIDATE_DIR / "baseline.jpg"
AFTER_PATH = CANDIDATE_DIR / "after.jpg"

for path in (BASELINE_PATH, AFTER_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

baseline_bytes = BASELINE_PATH.read_bytes()
after_bytes = AFTER_PATH.read_bytes()

client = boto3.client("bedrock-runtime", region_name=REGION)

prompt = """
You are reviewing a candidate region flagged by an image-change detection system.

Image 1: the baseline region before maintenance.
Image 2: the corresponding region after maintenance.

The region was flagged because the images contain a visual change.
Your task is to examine that change, not to repeat the detector's result blindly.

Perform these steps:

1. Describe the physical objects visible in Image 2. Include their shape,
   size relative to the crop, color, edges, and visible connections.
2. Compare Image 2 with Image 1. Identify which visible features are new,
   absent, moved, or changed.
3. Decide whether the evidence supports a newly introduced physical object,
   a change to an existing object, a lighting or shadow change, or uncertainty.
4. Explain the specific visual evidence for your decision.

Do not require an exact object classification. An object can be newly
introduced even if you cannot identify its function or name.
Do not assume a changed object is permitted or safe.
Do not report "no new object" solely because the object's purpose is unclear.

Return only one JSON object with these fields:
{
  "change_detected": "yes, no, or uncertain",
  "new_physical_object": "yes, no, or uncertain",
  "object_description": "description based on visible features",
  "evidence": ["specific differences between the images"],
  "possible_alternative_explanations": ["other plausible causes"],
  "confidence": "high, medium, or low",
  "limitations": ["relevant limitations"]
}
"""

try:
    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {"bytes": baseline_bytes},
                        }
                    },
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {"bytes": after_bytes},
                        }
                    },
                    {"text": prompt},
                ],
            }
        ],
        inferenceConfig={
            "maxTokens": 600,
            "temperature": 0,
        },
    )

    result = "\n".join(
        block["text"]
        for block in response["output"]["message"]["content"]
        if "text" in block
    )

    output_path = BASE_DIR / "bedrock_assessment.json"
    output_path.write_text(result + "\n", encoding="utf-8")

    print(result)
    print(f"\nAssessment saved to: {output_path}")

except ClientError as error:
    print("AWS error:", error.response["Error"]["Code"])
    print("Details:", error.response["Error"]["Message"])
