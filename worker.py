"""
Standalone worker process for a single NST job.

Invoked as: python worker.py <config.json path> <job_dir>

This file is launched via subprocess.Popen by JobManager.launch(). It runs in
a completely separate OS process and Python interpreter from the Streamlit
server. If it crashes, hangs, segfaults, or raises a CUDA OOM error, the
Streamlit server process is entirely unaffected -- JobManager simply observes
a dead PID or an "error" status.json and reports it to the user.

All heavy ML imports (torch, main.py's NST pipeline) happen ONLY in this
process, never in app.py / the Streamlit process.
"""
from __future__ import annotations

import gc
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_manager import JobStatus  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [worker] %(levelname)s: %(message)s",
)
logger = logging.getLogger("nst_worker")


def write_status(job_dir: Path, status: JobStatus) -> None:
    """Atomic write (write-to-temp + rename) so the Streamlit side never
    reads a half-written status.json."""
    tmp = job_dir / "status.json.tmp"
    tmp.write_text(json.dumps(asdict(status)))
    tmp.replace(job_dir / "status.json")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: worker.py <config.json> <job_dir>", file=sys.stderr)
        return 2

    config_path = Path(sys.argv[1])
    job_dir = Path(sys.argv[2])
    job_id = job_dir.name
    started_at = time.time()

    write_status(job_dir, JobStatus(job_id, "running", started_at=started_at))

    # ---- Load job config -------------------------------------------------
    try:
        params = json.loads(config_path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read job config: %s", exc)
        write_status(job_dir, JobStatus(job_id, "error", "Internal error: bad job configuration.", started_at=started_at))
        return 1

    # ---- Import heavy ML dependencies (isolated to this process) ---------
    try:
        import torch
        from main import generate_style_transfer
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to import NST modules: %s\n%s", exc, traceback.format_exc())
        write_status(job_dir, JobStatus(job_id, "error", "Internal error: NST modules failed to load.", started_at=started_at))
        return 1

    # GPU safety: start with a clean slate.
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()

    # torch.cuda.OutOfMemoryError only exists on newer torch versions; fall
    # back to plain RuntimeError (with message sniffing below) on older ones.
    OOMError = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)

    try:
        final_path = generate_style_transfer(
            content_path=params["content_path"],
            style_path=params["style_path"],
            output_name=job_id,
            config_name=params["config_name"],
            output_dir=str(job_dir.parent),
            checkpoint=params["checkpoint"],
            image_size=params.get("image_size"),
            alpha=params.get("alpha"),
            beta=params.get("beta"),
            gamma=params.get("gamma"),
            num_steps=params.get("num_steps"),
        )
        write_status(
            job_dir,
            JobStatus(job_id, "done", "Style transfer complete.", final_path=final_path, started_at=started_at),
        )
        logger.info("Job %s completed successfully.", job_id)

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        write_status(job_dir, JobStatus(job_id, "error", "A required file (image or checkpoint) was not found.", started_at=started_at))

    except PermissionError as exc:
        logger.error("Permission error: %s", exc)
        write_status(job_dir, JobStatus(job_id, "error", "Permission denied while writing output files.", started_at=started_at))

    except OOMError as exc:
        logger.error("CUDA out of memory: %s", exc)
        write_status(
            job_dir,
            JobStatus(job_id, "error", "GPU ran out of memory. Try a smaller image size, fewer steps, or restart the job.", started_at=started_at),
        )

    except ValueError as exc:
        logger.error("Value error: %s", exc)
        write_status(job_dir, JobStatus(job_id, "error", f"Invalid configuration: {exc}", started_at=started_at))

    except RuntimeError as exc:
        msg = str(exc)
        if "out of memory" in msg.lower() or "cuda" in msg.lower() and "memory" in msg.lower():
            logger.error("CUDA OOM (RuntimeError path): %s", msg)
            write_status(
                job_dir,
                JobStatus(job_id, "error", "GPU ran out of memory. Try a smaller image size, fewer steps, or restart the job.", started_at=started_at),
            )
        else:
            logger.error("Runtime error: %s\n%s", msg, traceback.format_exc())
            write_status(job_dir, JobStatus(job_id, "error", "An internal error occurred during style transfer.", started_at=started_at))

    except Exception as exc:  # noqa: BLE001 - last-resort catch-all; worker must never crash silently
        logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
        write_status(job_dir, JobStatus(job_id, "error", "An unexpected error occurred. Please try again.", started_at=started_at))

    finally:
        # GPU + memory safety: always release, even on failure/timeout-kill.
        try:
            if cuda_available:
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        gc.collect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
