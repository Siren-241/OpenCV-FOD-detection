import cv2
import numpy as np
from pathlib import Path

current_dir = Path.cwd()
print(f"Current working directory: {current_dir}")

BASELINE = Path("src/data/baseline.png")
AFTER = Path("src/data/after.png")
OUTPUT = Path("src/results")
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return image


baseline = load_image(BASELINE)
after = load_image(AFTER)

if baseline.shape != after.shape:
    raise ValueError("Images must have the same dimensions.")

baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

baseline_blur = cv2.GaussianBlur(baseline_gray, (5, 5), 0)
after_blur = cv2.GaussianBlur(after_gray, (5, 5), 0)

baseline_edges = cv2.Canny(baseline_blur, 60, 150)
after_edges = cv2.Canny(after_blur, 60, 150)

kernel = np.ones((3, 3), np.uint8)

baseline_tolerant = cv2.dilate(
    baseline_edges, kernel, iterations=2
)
after_tolerant = cv2.dilate(
    after_edges, kernel, iterations=2
)

baseline_only = cv2.bitwise_and(
    baseline_edges,
    cv2.bitwise_not(after_tolerant)
)

after_only = cv2.bitwise_and(
    after_edges,
    cv2.bitwise_not(baseline_tolerant)
)

difference = cv2.bitwise_or(baseline_only, after_only)
contours, _ = cv2.findContours(
    difference,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

result = after.copy()
detections = 0

for contour in contours:
    area = cv2.contourArea(contour)

    if area < 30:
        continue

    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
    detections += 1

cv2.imwrite(str(OUTPUT / "baseline_edges.jpg"), baseline_edges)
cv2.imwrite(str(OUTPUT / "after_edges.jpg"), after_edges)
cv2.imwrite(str(OUTPUT / "difference.jpg"), difference)
cv2.imwrite(str(OUTPUT / "detections.jpg"), result)

print(f"Candidate regions detected: {detections}")
print(f"Results saved to: {OUTPUT.resolve()}")

cv2.imshow("Baseline Edges", baseline_edges)
cv2.imshow("After Edges", after_edges)
cv2.imshow("Difference", difference)
cv2.imshow("Candidate Regions", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
