import cv2
import numpy as np
from pathlib import Path

BASELINE_PATH = Path("src/data/baseline.png")
AFTER_PATH = Path("src/data/after.png")
OUTPUT_DIR = Path("src/results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

baseline = cv2.imread(str(BASELINE_PATH))
after = cv2.imread(str(AFTER_PATH))

if baseline is None or after is None:
    raise FileNotFoundError(
        "Could not load baseline.jpg or after.jpg. Check data/."
    )

gray_baseline = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=3000)

kp1, des1 = orb.detectAndCompute(gray_baseline, None)
kp2, des2 = orb.detectAndCompute(gray_after, None)

print(f"Baseline features: {len(kp1)}")
print(f"After features: {len(kp2)}")

if des1 is None or des2 is None:
    raise RuntimeError("Not enough visual features to align these images.")

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
pairs = matcher.knnMatch(des1, des2, k=2)

good_matches = [
    m for pair in pairs
    if len(pair) == 2
    for m, n in [pair]
    if m.distance < 0.75 * n.distance
]

print(f"Good matches: {len(good_matches)}")

if len(good_matches) < 10:
    raise RuntimeError(
        "Too few reliable matches. Try images with more fixed visual detail."
    )

points1 = np.float32([
    kp1[m.queryIdx].pt for m in good_matches
]).reshape(-1, 1, 2)

points2 = np.float32([
    kp2[m.trainIdx].pt for m in good_matches
]).reshape(-1, 1, 2)

H, inlier_mask = cv2.findHomography(
    points2,
    points1,
    cv2.RANSAC,
    4.0
)

if H is None or inlier_mask is None:
    raise RuntimeError("Homography estimation failed.")

inliers = int(inlier_mask.sum())
inlier_ratio = inliers / len(good_matches)

print(f"RANSAC inliers: {inliers}")
print(f"Inlier ratio: {inlier_ratio:.2%}")

matches_image = cv2.drawMatches(
    baseline,
    kp1,
    after,
    kp2,
    good_matches,
    None,
    matchColor=(0, 255, 0),
    singlePointColor=(0, 0, 255),
    matchesMask=inlier_mask.ravel().tolist(),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    str(OUTPUT_DIR / "feature_matches.jpg"),
    matches_image
)

if inliers < 10 or inlier_ratio < 0.25:
    raise RuntimeError(
        "Alignment is unreliable. Inspect feature_matches.jpg "
        "before proceeding."
    )

height, width = gray_baseline.shape

aligned_after = cv2.warpPerspective(
    after,
    H,
    (width, height)
)

valid_after = cv2.warpPerspective(
    np.full(gray_after.shape, 255, dtype=np.uint8),
    H,
    (width, height)
)

cv2.imwrite(
    str(OUTPUT_DIR / "aligned_after.jpg"),
    aligned_after
)
cv2.imwrite(
    str(OUTPUT_DIR / "valid_after_mask.jpg"),
    valid_after
)

overlay = cv2.addWeighted(baseline, 0.5, aligned_after, 0.5, 0)

cv2.imwrite(
    str(OUTPUT_DIR / "alignment_overlay.jpg"),
    overlay
)

print("Alignment outputs saved to:", OUTPUT_DIR.resolve())

