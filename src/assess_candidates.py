import json
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REGION = "ap-southeast-2"
MODEL_ID = "amazon.nova-pro-v1:0"
MAX_AI_ASSESSMENTS = 10

ROOT = Path("src")
OUTPUT_DIR = ROOT / "results"
CROPS_DIR = OUTPUT_DIR / "candidates"

client = boto3.client("bedrock-runtime", region_name=REGION)

prompt = """
You are reviewing a candidate region flagged by an image-change detector.

Image 1 is the baseline region before maintenance.
Image 2 is the corresponding region after maintenance.

Describe the physical objects visible in Image 2. Compare both images
and identify features that are new, absent, moved, or changed.

Decide whether the evidence supports a newly introduced physical object,
a change to an existing object, a lighting or shadow change, or uncertainty.

Do not require an exact object classification. An object can be new even
when its function or name is unknown. Do not assume that every difference
is debris. Do not claim the inspected area is safe.

Return only one JSON object with these fields:
{
  "change_detected": "yes, no, or uncertain",
  "new_physical_object": "yes, no, or uncertain",
  "object_description": "visible features",
  "evidence": ["specific visual differences"],
  "possible_alternative_explanations": ["other plausible causes"],
  "confidence": "high, medium, or low",
  "limitations": ["relevant limitations"]
}
"""

regions_file = OUTPUT_DIR / "candidate_regions.json"

if not regions_file.is_file():
    raise FileNotFoundError(
        "Candidate metadata is missing. Run src/detect_changes.py first."
    )

candidates = json.loads(
    regions_file.read_text(encoding="utf-8")
)

if not isinstance(candidates, list):
    raise ValueError("Candidate metadata must contain a JSON list.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

selected_candidates = candidates[:MAX_AI_ASSESSMENTS]
assessments = []

print(f"Candidates detected: {len(candidates)}")
print(f"Maximum AI assessments: {MAX_AI_ASSESSMENTS}")
print(f"Candidates selected: {len(selected_candidates)}")

for candidate in selected_candidates:
    candidate_id = candidate["candidate_id"]
    baseline_path = ROOT.parent / candidate["baseline_crop"]
    after_path = ROOT.parent / candidate["after_crop"]

    if not baseline_path.is_file() or not after_path.is_file():
        raise FileNotFoundError(
            f"Missing paired crop for {candidate_id}: "
            f"{baseline_path}, {after_path}"
        )

    print(f"Assessing {candidate_id}...")

    record = {
        **candidate,
        "human_review_required": True,
        "model": MODEL_ID,
    }

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
                                "source": {
                                    "bytes": baseline_path.read_bytes()
                                },
                            }
                        },
                        {
                            "image": {
                                "format": "jpeg",
                                "source": {
                                    "bytes": after_path.read_bytes()
                                },
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

        text_output = "\n".join(
            block["text"]
            for block in response["output"]["message"]["content"]
            if "text" in block
        )

        try:
            start = text_output.index("{")
            end = text_output.rindex("}") + 1

            record["assessment"] = json.loads(
                text_output[start:end]
            )
        except (ValueError, json.JSONDecodeError):
            record["assessment"] = {
                "parse_error": True,
                "raw_response": text_output,
            }

    except ClientError as error:
        record["error"] = {
            "code": error.response["Error"]["Code"],
            "message": error.response["Error"]["Message"],
        }

    assessments.append(record)

    assessment_path = (
        OUTPUT_DIR / f"{candidate_id}_assessment.json"
    )
    assessment_path.write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )

report_path = OUTPUT_DIR / "fod_assessment_report.json"

report = {
    "model": MODEL_ID,
    "candidate_count": len(candidates),
    "assessed_count": len(assessments),
    "unassessed_count": len(candidates) - len(assessments),
    "max_ai_assessments": MAX_AI_ASSESSMENTS,
    "human_review_required": True,
    "assessments": assessments,
}

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print(f"\nCandidates detected: {len(candidates)}")
print(f"AI assessments completed: {len(assessments)}")
print(
    f"Candidates not assessed: "
    f"{len(candidates) - len(assessments)}"
)
print(f"Report saved to: {report_path.resolve()}")
