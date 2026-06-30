"""
Neural Style Transfer -- production-grade Streamlit front end.

SAFETY ARCHITECTURE (read this before touching the file)
----------------------------------------------------------------------------
1. PROCESS ISOLATION: the actual NST computation never runs inside this
   Streamlit process or a thread. JobManager (job_manager.py) launches it as
   a separate OS subprocess via worker.py. If the worker crashes, hangs, or
   OOMs, this script and the Streamlit server are completely unaffected.

2. SINGLE-FLIGHT CONCURRENCY: JobManager enforces a filesystem lock so only
   one job runs at a time across ALL sessions/tabs on this server.

3. TIMEOUT: any job running longer than MAX_JOB_TIME_SECONDS is forcibly
   killed (whole process group) and reported to the user.

4. UPLOAD VALIDATION + SANITIZATION: every uploaded file is checked for
   extension, size, and real image integrity (image_utils.py) before it ever
   touches disk in a usable form, and is re-encoded as a clean RGB JPEG.

5. NO ARBITRARY PATHS: there is no "output directory" text field anymore.
   All output lives under temp/jobs/<job_id>/, a server-generated path the
   user cannot influence -- this removes the path-traversal hole in the
   original app.

6. SESSION STATE: st.session_state replaces the original module-level
   mutable globals so job tracking is safe across Streamlit reruns.
"""
from __future__ import annotations

import logging
import logging.handlers
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
from PIL import Image

from image_utils import (
    ImageValidationError,
    MAX_DIMENSION,
    MAX_UPLOAD_BYTES,
    sanitize_image,
    validate_extension,
    validate_size,
)
from job_manager import (
    JobManager,
    JobStatus,
    MAX_JOB_TIME_SECONDS,
    UPLOAD_DIR,
    cleanup_old_files,
)

# ---------------------------------------------------------------------------
# Logging: rotating file handler so logs/app.log never grows unbounded.
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("nst_app")
if not logger.handlers:  # avoid duplicate handlers across Streamlit reruns
    logger.setLevel(logging.INFO)
    _handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "app.log", maxBytes=2_000_000, backupCount=5
    )
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Constants (server-controlled allow-lists -- never trust frontend values)
# ---------------------------------------------------------------------------
CONFIG_OPTIONS = ["default", "default_512", "custom", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"]
CHECKPOINT_OPTIONS = ["imagenette", "stl-10"]
IMAGE_SIZE_MIN, IMAGE_SIZE_MAX = 64, 1024
NUM_STEPS_MIN, NUM_STEPS_MAX = 1, 300
POLL_INTERVAL_SECONDS = 0.7


def clamp(value: float, lo: float, hi: float) -> float:
    """Defense in depth: even though sliders already constrain input, never
    trust that a value reaching this code path is actually in range."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# One-time setup per server process: stale-file cleanup + a shared JobManager
# (st.cache_resource makes this a singleton across all sessions, which is
# exactly what we want for the global single-job-at-a-time lock).
# ---------------------------------------------------------------------------
@st.cache_resource
def get_job_manager() -> JobManager:
    cleanup_old_files(max_age_hours=24)
    logger.info("JobManager initialized; stale file cleanup complete.")
    return JobManager()


jm = get_job_manager()

# ---------------------------------------------------------------------------
# Session state (replaces the original module-level mutable globals)
# ---------------------------------------------------------------------------
_session_defaults = {"job_id": None, "running": False, "job_started_at": 0.0}
for _key, _val in _session_defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_progress(progress_path: Path) -> Optional[Tuple[int, int, float]]:
    """Safely read progress.txt. The worker may be mid-write at any instant,
    so a partially-written or momentarily malformed file must never crash
    the UI -- just skip this poll and try again next time."""
    if not progress_path.exists():
        return None
    try:
        raw = progress_path.read_text().strip()
        step_s, total_s, loss_s = raw.split(",")
        return int(step_s), int(total_s), float(loss_s)
    except (OSError, ValueError):
        return None


def status_badge(state: str) -> None:
    badges = {
        "running": ("🔵 Running", st.info),
        "done": ("🟢 Done", st.success),
        "error": ("🔴 Error", st.error),
        "timeout": ("🟠 Timed out", st.warning),
        "cancelled": ("🟡 Cancelled", st.warning),
    }
    label, fn = badges.get(state, ("⚪ Unknown", st.info))
    fn(label)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Neural Style Transfer", layout="wide")
st.title("🎨 Neural Style Transfer")
st.write("Upload a content image and a style image, then generate a stylized result.")

# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------
content_file = st.file_uploader("Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Style Image", type=["jpg", "jpeg", "png"])
st.caption(
    f"Max file size: {MAX_UPLOAD_BYTES // (1024 * 1024)} MB · "
    f"Max resolution: {MAX_DIMENSION}x{MAX_DIMENSION} · Formats: jpg, jpeg, png"
)

# ---------------------------------------------------------------------------
# Sidebar: configuration
# (NOTE: the original free-text "Output Directory" / "Output Folder Name"
# fields have been REMOVED. They allowed arbitrary filesystem writes / path
# traversal. Output paths are now always server-generated job directories.)
# ---------------------------------------------------------------------------
image_size = alpha = beta = gamma = num_steps = None

with st.sidebar:
    st.header("Configuration")
    config_name = st.selectbox("Config", CONFIG_OPTIONS, index=0)

    if config_name == "custom":
        with st.expander("Custom Hyperparameters", expanded=True):
            image_size = st.slider(
                "Image Size", min_value=IMAGE_SIZE_MIN, max_value=IMAGE_SIZE_MAX, value=256, step=64
            )
            alpha = st.number_input("Content Weight (Alpha)", value=1.0, min_value=0.0)
            beta_exponent = st.slider("Style Weight (Beta) Exponent", 1, 10, 6)
            beta = 10 ** beta_exponent
            gamma_exponent = st.slider("TV Weight (Gamma) Exponent", -10, -1, -6)
            gamma = 10 ** gamma_exponent
            num_steps = st.slider("Optimization Steps", NUM_STEPS_MIN, NUM_STEPS_MAX, 50)

    checkpoint_name = st.selectbox("Checkpoint", CHECKPOINT_OPTIONS)

    st.caption("Outputs are stored internally per job; no manual paths needed.")

# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------
if content_file and style_file:
    _, col1, _, col2, _, col3, _ = st.columns([0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1])
    with col1:
        st.subheader("Content")
        st.image(content_file)
    with col2:
        st.subheader("Style")
        st.image(style_file)
    with col3:
        st.subheader("Generated")
        st.caption("Preview will appear here once a job is running.")

# ---------------------------------------------------------------------------
# Generate button
# ---------------------------------------------------------------------------
generate_clicked = st.button(
    "Generate Style Transfer",
    disabled=st.session_state.running,
    help="Disabled while a job is already running.",
)

if generate_clicked:
    if content_file is None or style_file is None:
        st.error("Please upload both images.")
        st.stop()

    # Concurrency control: refuse a second job even if the button somehow
    # gets pressed twice (double-click, slow rerun, another tab, etc).
    if jm.is_busy():
        st.warning("Another style transfer job is currently running. Please wait.")
        st.stop()

    # --- Upload validation + sanitization -----------------------------
    try:
        content_bytes = content_file.getvalue()
        style_bytes = style_file.getvalue()

        validate_extension(content_file.name)
        validate_extension(style_file.name)
        validate_size(content_bytes)
        validate_size(style_bytes)

        upload_id = uuid.uuid4().hex[:10]
        content_path = UPLOAD_DIR / f"{upload_id}_content.jpg"
        style_path = UPLOAD_DIR / f"{upload_id}_style.jpg"

        sanitize_image(content_bytes, content_path)
        sanitize_image(style_bytes, style_path)
        logger.info("Uploads validated and sanitized (upload_id=%s)", upload_id)

    except ImageValidationError as exc:
        st.error(f"Upload rejected: {exc}")
        logger.warning("Upload rejected: %s", exc)
        st.stop()
    except Exception as exc:  # noqa: BLE001 - never show a raw traceback to the user
        st.error("Could not process the uploaded images. Please try different files.")
        logger.error("Unexpected upload processing failure: %s", exc, exc_info=True)
        st.stop()

    # --- Build job params (server-side clamping; never trust the frontend) -
    if config_name == "custom":
        params = {
            "config_name": "custom",
            "checkpoint": checkpoint_name if checkpoint_name in CHECKPOINT_OPTIONS else "imagenette",
            "image_size": int(clamp(image_size, IMAGE_SIZE_MIN, IMAGE_SIZE_MAX)),
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
            "num_steps": int(clamp(num_steps, NUM_STEPS_MIN, NUM_STEPS_MAX)),
        }
    else:
        params = {
            "config_name": config_name if config_name in CONFIG_OPTIONS else "default",
            "checkpoint": checkpoint_name if checkpoint_name in CHECKPOINT_OPTIONS else "imagenette",
        }

    params["content_path"] = str(content_path)
    params["style_path"] = str(style_path)

    # --- Launch in an isolated subprocess ------------------------------
    job_id = jm.launch(params)
    if job_id is None:
        st.warning("Another style transfer job is currently running. Please wait.")
        st.stop()

    st.session_state.job_id = job_id
    st.session_state.running = True
    st.session_state.job_started_at = time.time()
    logger.info("Job %s launched (config=%s, checkpoint=%s)", job_id, params["config_name"], params["checkpoint"])
    st.rerun()

# ---------------------------------------------------------------------------
# Active job panel: polls status.json / progress.txt / latest.jpg.
# Uses an st.rerun()-driven poll loop (rather than a blocking while-loop) so
# the Cancel button stays interactive between polls.
# ---------------------------------------------------------------------------
if st.session_state.job_id:
    job_id = st.session_state.job_id
    st.divider()
    st.subheader("Live Stylization")

    status = jm.get_status(job_id)
    status_badge(status.state)

    elapsed = time.time() - st.session_state.job_started_at
    info_col, cancel_col = st.columns([4, 1])
    with info_col:
        st.caption(
            f"⏱️ Elapsed: {elapsed:.0f}s  ·  Timeout at {MAX_JOB_TIME_SECONDS // 60} min  ·  Job ID: {job_id}"
        )
    with cancel_col:
        if status.state == "running" and st.button("✖ Cancel Job"):
            jm.terminate(job_id, reason="cancelled")
            logger.info("Job %s cancelled by user.", job_id)
            st.session_state.job_id = None
            st.session_state.running = False
            st.warning("Job cancelled.")
            st.rerun()

    # Enforce the hard timeout regardless of what the worker is doing.
    if status.state == "running" and jm.check_timeout(job_id, MAX_JOB_TIME_SECONDS):
        status = jm.get_status(job_id)
        logger.warning("Job %s timed out.", job_id)

    progress_bar = st.progress(0)
    step_text = st.empty()
    loss_text = st.empty()
    generated_placeholder = st.empty()

    progress = read_progress(jm.progress_path(job_id))
    if progress:
        step, total_steps, loss = progress
        if total_steps > 0:
            progress_bar.progress(min(step / total_steps, 1.0))
        step_text.write(f"### Step {step}/{total_steps}")
        loss_text.write(f"### Loss: {loss:.4f}")

    latest_image = jm.latest_image_path(job_id)
    if latest_image.exists():
        try:
            img = Image.open(latest_image)
            img.load()
            generated_placeholder.image(img, caption="Live Output")
            img.close()
        except Exception:  # noqa: BLE001 - image may be mid-write; just skip this frame
            generated_placeholder.info("⏳ Updating image...")

    # --- Terminal states ------------------------------------------------
    if status.state == "running":
        time.sleep(POLL_INTERVAL_SECONDS)
        st.rerun()

    elif status.state == "done":
        jm.finish_and_release(job_id)
        st.session_state.running = False
        progress_bar.progress(1.0)
        step_text.write("### Completed")

        final_path = Path(status.final_path) if status.final_path else jm.job_directory(job_id) / "final_output.jpg"
        if final_path.exists():
            generated_placeholder.image(str(final_path), caption="Final Result")
            st.success("Style Transfer Complete!")
            with open(final_path, "rb") as f:
                st.download_button("Download Result", f, file_name="stylized.jpg")
            logger.info("Job %s delivered final image to user.", job_id)
        else:
            st.error("Job finished but the output image could not be found.")
            logger.error("Job %s marked done but final image missing.", job_id)

        st.session_state.job_id = None

    else:  # error, timeout, cancelled
        jm.finish_and_release(job_id)
        st.session_state.running = False
        if status.state == "timeout":
            st.error(f"⏱️ Job timed out after {MAX_JOB_TIME_SECONDS // 60} minutes and was terminated.")
        elif status.state == "cancelled":
            st.warning("Job was cancelled.")
        else:
            st.error(f"Style transfer failed: {status.message or 'Unknown error.'}")
        st.session_state.job_id = None
