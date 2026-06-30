"""
JobManager: process-isolated execution of Neural Style Transfer jobs.

Every NST run executes in its own OS subprocess (worker.py), never inside the
Streamlit process and never inside a thread. This is the core safety property
of the whole app: a CUDA OOM, a segfault, an infinite loop, or any bug in the
NST/model code can only kill the *worker* process. The Streamlit server keeps
running and JobManager simply observes a dead/failed subprocess and reports a
friendly error.

Concurrency is enforced with an on-disk lock file (atomic create), so only one
job runs at a time even across multiple browser sessions/tabs hitting the same
server -- this is intentionally NOT just an st.session_state flag, because
session_state is per-session and would not stop two different users from
launching jobs simultaneously.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nst_app")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "temp" / "uploads"
JOBS_DIR = BASE_DIR / "temp" / "jobs"
LOCK_FILE = JOBS_DIR / ".job.lock"

# Configurable timeout: a single job (worker process) is killed if it runs
# longer than this, regardless of what it's doing.
MAX_JOB_TIME_SECONDS = 15 * 60  # 15 minutes

for _d in (UPLOAD_DIR, JOBS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class JobStatus:
    job_id: str
    state: str  # "running" | "done" | "error" | "timeout" | "cancelled"
    message: str = ""
    final_path: Optional[str] = None
    started_at: float = 0.0


def _pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is still alive (POSIX)."""
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't own it -- still alive from our POV.
        return True
    return True


class JobManager:
    """Launches, tracks, terminates, and reports on NST worker subprocesses."""

    # ------------------------------------------------------------------
    # Concurrency lock (filesystem-based, survives Streamlit reruns and is
    # shared across all sessions hitting this server)
    # ------------------------------------------------------------------
    def _read_lock(self) -> Optional[dict]:
        if not LOCK_FILE.exists():
            return None
        try:
            return json.loads(LOCK_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def is_busy(self) -> Optional[str]:
        """Return the active job_id if a job is currently running, else None.

        Also self-heals: if the lock points at a PID that's no longer alive
        (e.g. the server crashed mid-job), the stale lock is removed so the
        app doesn't get permanently wedged in a "busy" state.
        """
        lock = self._read_lock()
        if lock is None:
            return None
        if _pid_alive(lock.get("pid", -1)):
            return lock.get("job_id")
        logger.warning("Removing stale lock file for dead pid %s", lock.get("pid"))
        self._release_lock()
        return None

    def _acquire_lock(self, job_id: str, pid: int) -> bool:
        """Atomically create the lock file (O_EXCL). False if already locked."""
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            json.dump({"job_id": job_id, "pid": pid, "started_at": time.time()}, f)
        return True

    def _release_lock(self) -> None:
        try:
            LOCK_FILE.unlink(missing_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Launching
    # ------------------------------------------------------------------
    def launch(self, params: dict) -> Optional[str]:
        """
        Start a new NST job in an isolated subprocess.
        Returns the new job_id, or None if another job is already running
        (caller should show "another job is running" and not retry).
        """
        if self.is_busy():
            return None

        job_id = uuid.uuid4().hex[:12]
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        config_path = job_dir / "config.json"
        config_path.write_text(json.dumps(params))

        started_at = time.time()
        status = JobStatus(job_id=job_id, state="running", started_at=started_at)
        (job_dir / "status.json").write_text(json.dumps(asdict(status)))

        log_path = job_dir / "worker.log"
        worker_script = BASE_DIR / "worker.py"

        # start_new_session=True puts the worker in its own process group, so
        # we can reliably kill it -- and any children/threads it spawns (e.g.
        # CUDA worker threads) -- on timeout or user cancellation.
        with open(log_path, "wb") as logfile:
            proc = subprocess.Popen(
                [sys.executable, str(worker_script), str(config_path), str(job_dir)],
                stdout=logfile,
                stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR),
                start_new_session=True,
            )

        if not self._acquire_lock(job_id, proc.pid):
            # Lost a race to another session between is_busy() and here.
            self._kill_process_group(proc.pid)
            shutil.rmtree(job_dir, ignore_errors=True)
            return None

        logger.info("Launched NST job %s (pid=%s)", job_id, proc.pid)
        return job_id

    # ------------------------------------------------------------------
    # Polling / status
    # ------------------------------------------------------------------
    def get_status(self, job_id: str) -> JobStatus:
        """Read status.json defensively -- the worker may be mid-write."""
        status_file = JOBS_DIR / job_id / "status.json"
        if not status_file.exists():
            return JobStatus(job_id=job_id, state="error", message="Job record missing.")
        for _ in range(3):
            try:
                data = json.loads(status_file.read_text())
                return JobStatus(**data)
            except (json.JSONDecodeError, OSError):
                time.sleep(0.05)  # brief retry covers a torn read mid-write
        return JobStatus(job_id=job_id, state="running", message="Reading status...")

    def check_timeout(self, job_id: str, max_seconds: int = MAX_JOB_TIME_SECONDS) -> bool:
        """If the job has run past max_seconds, kill it and mark it timed out."""
        lock = self._read_lock()
        if lock is None or lock.get("job_id") != job_id:
            return False
        elapsed = time.time() - lock.get("started_at", time.time())
        if elapsed > max_seconds:
            logger.warning("Job %s exceeded timeout (%.0fs); terminating.", job_id, elapsed)
            self.terminate(job_id, reason="timeout")
            return True
        return False

    # ------------------------------------------------------------------
    # Termination / cleanup
    # ------------------------------------------------------------------
    def _kill_process_group(self, pid: int) -> None:
        """SIGTERM, give it a moment, then SIGKILL the whole process group."""
        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, OSError):
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(1.0)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def terminate(self, job_id: str, reason: str = "cancelled") -> None:
        """Forcefully kill the worker (used for both Cancel button and timeout)."""
        lock = self._read_lock()
        if lock and lock.get("job_id") == job_id:
            self._kill_process_group(lock.get("pid", -1))
        self._release_lock()

        job_dir = JOBS_DIR / job_id
        status = JobStatus(job_id=job_id, state=reason, message=f"Job {reason}.")
        try:
            (job_dir / "status.json").write_text(json.dumps(asdict(status)))
        except OSError:
            pass
        logger.info("Job %s terminated (%s)", job_id, reason)

    def finish_and_release(self, job_id: str) -> None:
        """Call once a job reaches a terminal state, to free the global lock
        for the next job. Safe to call even if already released."""
        lock = self._read_lock()
        if lock and lock.get("job_id") == job_id:
            self._release_lock()

    @staticmethod
    def progress_path(job_id: str) -> Path:
        return JOBS_DIR / job_id / "progress.txt"

    @staticmethod
    def latest_image_path(job_id: str) -> Path:
        return JOBS_DIR / job_id / "latest.jpg"

    @staticmethod
    def job_directory(job_id: str) -> Path:
        return JOBS_DIR / job_id


def cleanup_old_files(max_age_hours: int = 24) -> None:
    """Delete uploads and job directories older than max_age_hours.

    Run once per server lifetime (see app.py's st.cache_resource wrapper) so
    abandoned jobs, failed-job artifacts, and orphaned uploads don't pile up
    on disk indefinitely.
    """
    cutoff = time.time() - max_age_hours * 3600
    for directory in (UPLOAD_DIR, JOBS_DIR):
        if not directory.exists():
            continue
        for entry in directory.iterdir():
            try:
                if entry.name.startswith("."):
                    continue  # don't touch the lock file
                if entry.stat().st_mtime < cutoff:
                    if entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        entry.unlink(missing_ok=True)
                    logger.info("Cleaned up stale file/dir: %s", entry)
            except OSError as exc:
                logger.warning("Cleanup skipped %s: %s", entry, exc)
                continue
