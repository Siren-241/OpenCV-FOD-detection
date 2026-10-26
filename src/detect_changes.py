import cv2
import numpy as np
import json
import shutil
from pathlib import Path

BASELINE_PATH = Path("src/data/baseline.png")
ALIGNED_PATH = Path("src/results/aligned_after.jpg")
MASK_PATH = Path("src/results/valid_after_mask.jpg")
OUTPUT_DIR = Path("src/results")
CROPS_DIR = OUTPUT_DIR / "candidates"

DIFF_THRESHOLD = 35
MIN_COMPONENT_AREA = 250
MIN_BOX_WIDTH = 20
MIN_BOX_HEIGHT = 20
MIN_CHANGED_PIXELS = 150
MIN_MEAN_DIFFERENCE = 18
PADDING = 20
MERGE_KERNEL_SIZE = 21

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

baseline = cv2.imread(str(BASELINE_PATH))
aligned = cv2.imread(str(ALIGNED_PATH))
valid_mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

if baseline is None or aligned is None or valid_mask is None:
    raise FileNotFoundError(
        "Missing input. Run src/align_images.py first and check the data/ and results/ paths."
    )

if baseline.shape != aligned.shape or baseline.shape[:2] != valid_mask.shape:
    raise ValueError("Image dimensions do not match.")

if CROPS_DIR.exists():
    shutil.rmtree(CROPS_DIR)

CROPS_DIR.mkdir(parents=True)

gray1 = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

valid_mask = cv2.threshold(
    valid_mask, 127, 255, cv2.THRESH_BINARY
)[1]

valid_mask = cv2.erode(
    valid_mask,
    cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
    iterations=1
)

difference = cv2.absdiff(gray1, gray2)
difference[valid_mask == 0] = 0

thresholded = cv2.threshold(
    difference, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY
)[1]

thresholded[valid_mask == 0] = 0

small_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)
)

cleaned = cv2.morphologyEx(
    thresholded, cv2.MORPH_OPEN, small_kernel, iterations=2
)

merge_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (MERGE_KERNEL_SIZE, MERGE_KERNEL_SIZE)
)

merged = cv2.morphologyEx(
    cleaned, cv2.MORPH_CLOSE, merge_kernel, iterations=1
)

merged[valid_mask == 0] = 0

contours, _ = cv2.findContours(
    merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

height, width = aligned.shape[:2]
result = aligned.copy()
candidates = []

for contour in sorted(contours, key=cv2.contourArea, reverse=True):
    area = float(cv2.contourArea(contour))

    if area < MIN_COMPONENT_AREA:
        continue

    x, y, w, h = cv2.boundingRect(contour)

    if w < MIN_BOX_WIDTH or h < MIN_BOX_HEIGHT:
        continue

    x1 = max(0, x - PADDING)
    y1 = max(0, y - PADDING)
    x2 = min(width, x + w + PADDING)
    y2 = min(height, y + h + PADDING)

    region_difference = difference[y:y + h, x:x + w]
    region_mask = merged[y:y + h, x:x + w] > 0

    changed_values = region_difference[region_mask]

    if changed_values.size == 0:
        continue

    changed_pixels = int(
        np.count_nonzero(
            (region_difference >= DIFF_THRESHOLD) & region_mask
        )
    )

    mean_difference = float(np.mean(changed_values))

    if changed_pixels < MIN_CHANGED_PIXELS:
        continue

    if mean_difference < MIN_MEAN_DIFFERENCE:
        continue

    baseline_crop = baseline[y1:y2, x1:x2]
    after_crop = aligned[y1:y2, x1:x2]

    if baseline_crop.size == 0 or after_crop.size == 0:
        continue

    candidate_id = len(candidates) + 1
    candidate_name = f"candidate_{candidate_id:03d}"
    candidate_dir = CROPS_DIR / candidate_name
    candidate_dir.mkdir(parents=True)

    baseline_file = candidate_dir / "baseline.jpg"
    after_file = candidate_dir / "after.jpg"

    if not cv2.imwrite(str(baseline_file), baseline_crop):
        raise RuntimeError(f"Could not save {baseline_file}")

    if not cv2.imwrite(str(after_file), after_crop):
        raise RuntimeError(f"Could not save {after_file}")

    candidates.append({
        "candidate_id": candidate_name,
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "contour_area": area,
        "changed_pixels": changed_pixels,
        "mean_difference": round(mean_difference, 2),
        "baseline_crop": str(baseline_file),
        "after_crop": str(after_file)
    })

    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.putText(
        result,
        f"Candidate {candidate_id}",
        (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )

cv2.imwrite(str(OUTPUT_DIR / "intensity_difference.jpg"), difference)
cv2.imwrite(str(OUTPUT_DIR / "thresholded_difference.jpg"), thresholded)
cv2.imwrite(str(OUTPUT_DIR / "cleaned_difference.jpg"), cleaned)
cv2.imwrite(str(OUTPUT_DIR / "merged_candidates_mask.jpg"), merged)
cv2.imwrite(str(OUTPUT_DIR / "change_candidates.jpg"), result)

(OUTPUT_DIR / "candidate_regions.json").write_text(
    json.dumps(candidates, indent=2),
    encoding="utf-8"
)

print(f"Candidate regions: {len(candidates)}")
print(f"Paired crops saved to: {CROPS_DIR.resolve()}")
print(f"Annotated image: {(OUTPUT_DIR / 'change_candidates.jpg').resolve()}")

