import json
import subprocess
import sys
from pathlib import Path

from io import BytesIO
from PIL import Image, UnidentifiedImageError

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "src" / "results"
REPORT_PATH = RESULTS_DIR / "fod_assessment_report.json"
REVIEWS_PATH = RESULTS_DIR / "human_reviews.json"

DATA_DIR = ROOT / "src" / "data"
BASELINE_PATH = DATA_DIR / "baseline.png"
AFTER_PATH = DATA_DIR / "after.png"
CAPTURE_SCRIPT = ROOT / "src" / "capture_images.py"

PIPELINE = [
    ("Image alignment", ROOT / "src" / "align_images.py"),
    ("Change detection", ROOT / "src" / "detect_changes.py"),
    ("AI assessment", ROOT / "src" / "assess_candidates.py"),
]

REVIEW_OPTIONS = [
    "Not reviewed",
    "Confirmed concern",
    "No concern identified",
    "Needs further inspection",
]


def load_json(path, default):
    if not path.is_file():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def save_review(candidate_id, decision, notes):
    reviews = load_json(REVIEWS_PATH, {})

    reviews[candidate_id] = {
        "decision": decision,
        "notes": notes,
    }

    REVIEWS_PATH.write_text(
        json.dumps(reviews, indent=2),
        encoding="utf-8",
    )

def capture_image(stage):
    if not CAPTURE_SCRIPT.is_file():
        return False, f"Capture script not found: {CAPTURE_SCRIPT}"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [sys.executable, str(CAPTURE_SCRIPT), stage],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False, "Camera capture timed out after 30 seconds."

    output = "\n".join(
        part for part in [result.stdout, result.stderr] if part
    )

    if result.returncode != 0:
        return False, output or f"Failed to capture {stage} image."

    image_path = BASELINE_PATH if stage == "baseline" else AFTER_PATH

    if not image_path.is_file():
        return False, f"Capture completed, but image was not created: {image_path}"

    return True, output or f"Captured {stage} image."

def uploaded_image_to_png(uploaded_file):
    """Validate an uploaded image and convert it to PNG bytes."""
    image_bytes = uploaded_file.getvalue()

    with Image.open(BytesIO(image_bytes)) as image:
        image.load()
        image = image.convert("RGB")

        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

def run_pipeline():
    if not BASELINE_PATH.is_file():
        return False, [], "Capture or provide the baseline image first."

    if not AFTER_PATH.is_file():
        return False, [], "Capture or provide the after-maintenance image first."
    logs = []

    for stage, script in PIPELINE:
        if not script.is_file():
            return False, logs, f"Script not found: {script}"

        with st.status(f"Running {stage}...", expanded=True) as status:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            output = "\n".join(
                part for part in [result.stdout, result.stderr] if part
            )

            logs.append({
                "stage": stage,
                "returncode": result.returncode,
                "output": output,
            })

            if result.returncode != 0:
                status.update(label=f"{stage} failed", state="error")
                return False, logs, f"{stage} failed."

            status.update(label=f"{stage} completed", state="complete")

    if not REPORT_PATH.is_file():
        return False, logs, "The assessment report was not created."

    return True, logs, None


st.set_page_config(
    page_title="FOD Detection Review",
    page_icon="🔎",
    layout="wide",
)

st.title("Foreign Object Debris Detection")
st.caption("Visual change detection with AI-assisted evidence review")

st.warning(
    "This is a research prototype. A negative result does not establish "
    "that an engine is safe. All candidates require human review."
)

st.subheader("Image input")

input_mode = st.radio(
    "Choose how to provide images",
    [
        "Upload images",
        "Android IP Camera (local network)",
    ],
    horizontal=True,
)

if input_mode == "Upload images":
    st.caption(
        "Recommended for the deployed demo. Upload the scene before "
        "maintenance and the scene after maintenance."
    )

    baseline_col, after_col = st.columns(2)

    with baseline_col:
        baseline_upload = st.file_uploader(
            "Baseline image",
            type=["png", "jpg", "jpeg", "webp"],
            key="baseline_upload",
        )

        if baseline_upload is not None:
            st.image(
                baseline_upload,
                caption="Selected baseline",
                use_container_width=True,
            )

    with after_col:
        after_upload = st.file_uploader(
            "After-maintenance image",
            type=["png", "jpg", "jpeg", "webp"],
            key="after_upload",
        )

        if after_upload is not None:
            st.image(
                after_upload,
                caption="Selected after-maintenance image",
                use_container_width=True,
            )

    if st.button(
        "Use uploaded images",
        type="primary",
        use_container_width=True,
        disabled=(
            baseline_upload is None or after_upload is None
        ),
    ):
        try:
            # Validate both images before replacing either saved image.
            baseline_png = uploaded_image_to_png(baseline_upload)
            after_png = uploaded_image_to_png(after_upload)

            DATA_DIR.mkdir(parents=True, exist_ok=True)

            BASELINE_PATH.write_bytes(baseline_png)
            AFTER_PATH.write_bytes(after_png)

            st.success("Both images are ready for processing.")

        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(f"Could not process the uploaded images: {exc}")

else:
    st.caption(
        "This mode requires the camera to be reachable from the "
        "machine running this application. It is intended for local use."
    )

    baseline_col, after_col = st.columns(2)

    with baseline_col:
        if st.button(
            "Capture baseline",
            use_container_width=True,
        ):
            success, message = capture_image("baseline")

            if success:
                st.success("Baseline image captured.")
                st.code(message)
            else:
                st.error(message)

    with after_col:
        if st.button(
            "Capture after maintenance",
            use_container_width=True,
        ):
            if not BASELINE_PATH.is_file():
                st.error("Capture the baseline image first.")
            else:
                success, message = capture_image("after")

                if success:
                    st.success("After-maintenance image captured.")
                    st.code(message)
                else:
                    st.error(message)

st.divider()
st.subheader("Images ready for analysis")

preview_col1, preview_col2 = st.columns(2)

with preview_col1:
    if BASELINE_PATH.is_file():
        st.markdown("**Baseline image**")
        st.image(
            str(BASELINE_PATH),
            use_container_width=True,
        )
    else:
        st.info("No baseline image is available.")

with preview_col2:
    if AFTER_PATH.is_file():
        st.markdown("**After-maintenance image**")
        st.image(
            str(AFTER_PATH),
            use_container_width=True,
        )
    else:
        st.info("No after-maintenance image is available.")

if st.button("Run detection pipeline", type="primary", use_container_width=True):
    with st.spinner("Processing images and assessing candidates..."):
        success, logs, error = run_pipeline()

    for log in logs:
        with st.expander(
            f"{log['stage']} — "
            f"{'Success' if log['returncode'] == 0 else 'Failed'}"
        ):
            st.code(log["output"] or "No output.")

    if success:
        st.success("Pipeline completed. Review the candidates below.")
        st.rerun()
    else:
        st.error(error or "Pipeline failed.")
        st.stop()

report = load_json(REPORT_PATH, None)

if report is None:
    st.info("Run the pipeline to generate your first assessment report.")
    st.stop()

candidates = report.get("assessments", [])
reviews = load_json(REVIEWS_PATH, {})

reviewed_count = sum(
    1
    for candidate in candidates
    if reviews.get(candidate.get("candidate_id"), {}).get(
        "decision", "Not reviewed"
    ) != "Not reviewed"
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Candidates detected",
    report.get("candidate_count", len(candidates)),
)
col2.metric(
    "AI assessments",
    report.get("assessed_count", len(candidates)),
)
col3.metric(
    "Not assessed",
    report.get(
        "unassessed_count",
        max(0, report.get("candidate_count", len(candidates))
            - len(candidates)),
    ),
)
col4.metric(
    "Reviewed",
    reviewed_count,
)

if report.get("unassessed_count", 0) > 0:
    st.warning(
        f"{report['unassessed_count']} detected candidates were not "
        "assessed by AI. They remain unresolved and require inspection."
    )

st.divider()
st.subheader("Candidate review")

if not candidates:
    st.info("No candidate regions were detected in this run.")

for candidate in candidates:
    candidate_id = candidate.get("candidate_id", "unknown")
    assessment = candidate.get("assessment", {})
    saved_review = reviews.get(candidate_id, {})
    current_decision = saved_review.get("decision", "Not reviewed")
    current_notes = saved_review.get("notes", "")

    with st.expander(
        f"{candidate_id} · "
        f"{assessment.get('change_detected', 'Assessment unavailable')} change",
        expanded=True,
    ):
        left, right = st.columns(2)

        baseline_path = ROOT / candidate.get("baseline_crop", "")
        after_path = ROOT / candidate.get("after_crop", "")

        with left:
            st.markdown("**Before maintenance**")
            if baseline_path.is_file():
                st.image(str(baseline_path), use_container_width=True)
            else:
                st.error("Baseline crop not found.")

        with right:
            st.markdown("**After maintenance**")
            if after_path.is_file():
                st.image(str(after_path), use_container_width=True)
            else:
                st.error("After crop not found.")

        st.markdown("**AI assessment**")

        if "error" in candidate:
            st.error(
                candidate["error"].get(
                    "message", "The AI assessment failed."
                )
            )
        elif assessment.get("parse_error"):
            st.warning("The model response could not be parsed as JSON.")
            st.code(assessment.get("raw_response", ""))
        elif assessment:
            st.write(
                "**New physical object:**",
                assessment.get("new_physical_object", "uncertain"),
            )
            st.write(
                "**Change detected:**",
                assessment.get("change_detected", "uncertain"),
            )
            st.write(
                "**Description:**",
                assessment.get("object_description", "Not provided"),
            )
            st.write(
                "**Confidence:**",
                assessment.get("confidence", "Not provided"),
            )

            st.markdown("**Evidence**")
            evidence = assessment.get("evidence", [])
            if evidence:
                for item in evidence:
                    st.markdown(f"- {item}")
            else:
                st.write("No evidence details provided.")

            alternatives = assessment.get(
                "possible_alternative_explanations", []
            )
            if alternatives:
                st.markdown("**Alternative explanations**")
                for item in alternatives:
                    st.markdown(f"- {item}")

            limitations = assessment.get("limitations", [])
            if limitations:
                st.markdown("**Limitations**")
                for item in limitations:
                    st.markdown(f"- {item}")
        else:
            st.warning("No assessment was returned.")

        st.divider()
        st.markdown("**Human review**")

        selected_decision = st.selectbox(
            "Review decision",
            REVIEW_OPTIONS,
            index=(
                REVIEW_OPTIONS.index(current_decision)
                if current_decision in REVIEW_OPTIONS
                else 0
            ),
            key=f"decision_{candidate_id}",
        )

        selected_notes = st.text_area(
            "Reviewer notes",
            value=current_notes,
            key=f"notes_{candidate_id}",
            placeholder="Record the reason for your decision.",
        )

        if st.button("Save review", key=f"save_{candidate_id}"):
            save_review(
                candidate_id,
                selected_decision,
                selected_notes,
            )
            st.success(f"Review saved for {candidate_id}.")

st.caption(
    "AI assessments support the review process. They do not replace "
    "qualified inspection or establish airworthiness."
)
