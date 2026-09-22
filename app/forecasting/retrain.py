"""
app/forecasting/retrain.py
=============================
Stage 2 of the retraining plan (Stage 1 = scfi_model_v3.py now pulling
live Supabase data via feature_engineering.py instead of a frozen CSV -
see that script's Section 0 for the rationale).

This module runs the (slow, ~30min-2hr) training script as a subprocess
into an ISOLATED location, compares its resulting CV metrics against
whatever's currently deployed, and only swaps the new artifacts into the
live MODEL_ARTIFACTS_DIR if they're equal-or-better - backing up the old
ones first. If worse, the old artifacts are left untouched and a warning
is logged. Never retrains silently into production: every swap is
preceded by a timestamped backup of exactly what was replaced, and if the
swap itself or the post-swap registry.reload() fails (e.g. a corrupt
artifact that copies fine but won't load), that same backup is copied
back over the live artifacts automatically so the app is never left
serving from a broken swap. Every code path - success, rejection, or any
failure - always ends by writing a terminal status to retrain_status.json;
it never gets stuck on "running".

Where "staging" actually is: scfi_model_v3.py's own SCRIPT_DIR (wherever
the file itself lives - the repo root) is where it saves
model_artifacts/ and SCFI_model_metrics.json by ITS OWN hardcoded logic.
Since the repo root's model_artifacts/ is a different directory from the
live backend/model_artifacts/ the running app actually reads from, the
repo root already acts as a natural staging area with zero changes
needed to the training script's own save logic.
"""

import asyncio
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from app.config import MODEL_ARTIFACTS_DIR as _CONFIGURED_MODEL_ARTIFACTS_DIR
from app.services.model_registry import registry
from app.services.supabase_service import supabase_service

logger = logging.getLogger("freightai.retrain")

# MODEL_ARTIFACTS_DIR in .env is set as a RELATIVE path ("./model_artifacts"),
# which is fine for the app itself (always launched with cwd=backend/), but
# .parent/.parent.parent arithmetic on a relative Path silently stops at "."
# instead of walking up to the real backend/repo-root directories. Resolve
# to absolute first so BACKEND_DIR/REPO_ROOT below are actually correct
# regardless of the process's cwd at import time.
MODEL_ARTIFACTS_DIR = _CONFIGURED_MODEL_ARTIFACTS_DIR.resolve()
BACKEND_DIR = MODEL_ARTIFACTS_DIR.parent
REPO_ROOT = BACKEND_DIR.parent

TRAINING_SCRIPT = REPO_ROOT / "scfi_model_v3.py"
TRAINING_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

STAGING_ARTIFACTS_DIR = REPO_ROOT / "model_artifacts"
STAGING_METRICS_PATH = REPO_ROOT / "SCFI_model_metrics.json"

LIVE_METRICS_PATH = BACKEND_DIR / "SCFI_model_metrics.json"
BACKUPS_ROOT = BACKEND_DIR / "model_artifacts_backups"

STATUS_PATH = BACKEND_DIR / "retrain_status.json"
TRAINING_LOG_PATH = BACKEND_DIR / "retrain_last_run.log"

RETRAIN_LOCK_NAME = "retraining"
RETRAIN_LOCK_TTL_SECONDS = 3 * 60 * 60  # 3h - generous for a real training run
PRIMARY_MODEL_FOR_COMPARISON = "ensemble"


def _write_status(**fields) -> None:
    payload = {"updated_at": datetime.utcnow().isoformat(), **fields}
    STATUS_PATH.write_text(json.dumps(payload, indent=2))


def read_status() -> dict:
    if not STATUS_PATH.exists():
        return {"status": "never_run"}
    return json.loads(STATUS_PATH.read_text())


def _load_metrics(path: Path) -> "dict | None":
    if not path.exists():
        return None
    return json.loads(path.read_text())


def compare_metrics(new_metrics: dict, old_metrics: dict, primary_model: str = PRIMARY_MODEL_FOR_COMPARISON) -> dict:
    """Lower MAE wins. Missing data on either side fails closed (treated
    as NOT better, so an incomplete run never gets swapped in by
    accident)."""
    def _mae_for(metrics: dict) -> "float | None":
        for m in metrics.get("models", []):
            if m.get("id") == primary_model:
                return m.get("MAE")
        return None

    new_mae = _mae_for(new_metrics)
    old_mae = _mae_for(old_metrics)

    if new_mae is None or old_mae is None:
        return {
            "better_or_equal": False, "reason": "missing_mae",
            "primary_model": primary_model, "new_mae": new_mae, "old_mae": old_mae,
        }

    return {
        "better_or_equal": new_mae <= old_mae,
        "primary_model": primary_model, "new_mae": new_mae, "old_mae": old_mae,
    }


def backup_live_artifacts() -> Path:
    """Copies the CURRENT live model_artifacts/ + SCFI_model_metrics.json
    into a fresh timestamped folder before anything gets overwritten."""
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_dir = BACKUPS_ROOT / stamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    if MODEL_ARTIFACTS_DIR.exists():
        shutil.copytree(MODEL_ARTIFACTS_DIR, backup_dir / "model_artifacts", dirs_exist_ok=True)
    if LIVE_METRICS_PATH.exists():
        shutil.copy2(LIVE_METRICS_PATH, backup_dir / "SCFI_model_metrics.json")

    logger.info("Backed up live artifacts to %s", backup_dir)
    return backup_dir


def swap_in_staged_artifacts() -> None:
    """Copies the freshly-trained staging artifacts over the live ones.
    Only ever called AFTER backup_live_artifacts() and a favorable
    compare_metrics() result."""
    for item in STAGING_ARTIFACTS_DIR.iterdir():
        dest = MODEL_ARTIFACTS_DIR / item.name
        if item.is_file():
            shutil.copy2(item, dest)
    shutil.copy2(STAGING_METRICS_PATH, LIVE_METRICS_PATH)
    logger.info("Swapped staged artifacts into %s", MODEL_ARTIFACTS_DIR)


def restore_live_artifacts_from_backup(backup_dir: Path) -> None:
    """Undoes swap_in_staged_artifacts() by copying a backup_live_artifacts()
    snapshot back over the live model_artifacts/ + SCFI_model_metrics.json.
    Used when the newly-swapped-in artifacts turn out to be broken (e.g. a
    corrupt .keras file that swapped fine but fails to load), so a failed
    retrain never leaves the live backend stuck on disk artifacts nobody
    can successfully load."""
    backed_up_artifacts = backup_dir / "model_artifacts"
    if backed_up_artifacts.exists():
        for item in backed_up_artifacts.iterdir():
            if item.is_file():
                shutil.copy2(item, MODEL_ARTIFACTS_DIR / item.name)
    backed_up_metrics = backup_dir / "SCFI_model_metrics.json"
    if backed_up_metrics.exists():
        shutil.copy2(backed_up_metrics, LIVE_METRICS_PATH)
    logger.info("Restored live artifacts from backup %s", backup_dir)


async def _run_training_subprocess() -> tuple[int, str]:
    """Runs scfi_model_v3.py in the SAME dedicated venv it needs (heavy
    deps: tensorflow, xgboost, optuna, shap, statsmodels - deliberately
    separate from the lightweight FastAPI backend's own venv). Output is
    streamed to a log file since a real run can produce a lot of it over
    a long time."""
    if not TRAINING_PYTHON.exists():
        raise FileNotFoundError(
            f"Training venv not found at {TRAINING_PYTHON} - see docs/PROJECT_DOCUMENTATION.md "
            "for the repo-root .venv this script needs (tensorflow/xgboost/optuna/shap/statsmodels)."
        )

    # scfi_model_v3.py prints unicode characters (e.g. "->"/checkmarks).
    # With stdout redirected to a file (not a real console), Windows
    # subprocesses default to the system codepage (cp1252 here) instead of
    # UTF-8, which can't encode them and crashes the run before it does
    # any real work. Forcing PYTHONIOENCODING fixes this without touching
    # the training script itself.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    with open(TRAINING_LOG_PATH, "w", encoding="utf-8") as log_file:
        process = await asyncio.create_subprocess_exec(
            str(TRAINING_PYTHON), str(TRAINING_SCRIPT),
            cwd=str(REPO_ROOT), env=env,
            stdout=log_file, stderr=asyncio.subprocess.STDOUT,
        )
        returncode = await process.wait()
    return returncode, str(TRAINING_LOG_PATH)


async def run_retraining_and_maybe_swap(triggered_by: str) -> dict:
    """
    Full Stage 2 flow: acquire lock -> run training subprocess (slow) ->
    compare new vs. live metrics -> backup + swap + registry.reload() if
    better-or-equal, else leave live artifacts untouched -> release lock.
    """
    if not await supabase_service.try_acquire_lock(RETRAIN_LOCK_NAME, triggered_by, RETRAIN_LOCK_TTL_SECONDS):
        return {"status": "skipped", "reason": "retraining already in progress elsewhere"}

    _write_status(status="running", triggered_by=triggered_by, started_at=datetime.utcnow().isoformat())
    try:
        returncode, log_path = await _run_training_subprocess()
        if returncode != 0:
            result = {"status": "failed", "reason": f"training subprocess exited {returncode}", "log_path": log_path}
            _write_status(**result)
            logger.error("Retraining subprocess failed (exit %s) - see %s", returncode, log_path)
            return result

        new_metrics = _load_metrics(STAGING_METRICS_PATH)
        old_metrics = _load_metrics(LIVE_METRICS_PATH)
        if new_metrics is None:
            result = {"status": "failed", "reason": "training completed but produced no metrics JSON"}
            _write_status(**result)
            return result
        if old_metrics is None:
            # Nothing currently deployed to compare against - treat any
            # successful run as an improvement over "no model at all".
            comparison = {"better_or_equal": True, "reason": "no_live_metrics_to_compare"}
        else:
            comparison = compare_metrics(new_metrics, old_metrics)

        if comparison["better_or_equal"]:
            backup_dir = backup_live_artifacts()
            try:
                swap_in_staged_artifacts()
                registry.reload()
            except Exception as exc:
                # Artifacts may already be half- or fully-swapped on disk at
                # this point (swap_in_staged_artifacts succeeded but reload()
                # failed to load them, e.g. a corrupt .keras file) - roll
                # disk back to the pre-swap backup so the live app is never
                # left pointed at artifacts nobody can load, then reload
                # again so the in-memory registry matches disk once more.
                logger.exception(
                    "Swap/reload failed after backing up live artifacts to %s - rolling back.",
                    backup_dir,
                )
                try:
                    restore_live_artifacts_from_backup(backup_dir)
                    registry.reload()
                    rollback_note = "rolled back to previous artifacts successfully"
                except Exception as rollback_exc:
                    logger.critical(
                        "Rollback after failed swap ALSO failed (%s) - live artifacts may be "
                        "inconsistent with the running registry. Manual intervention needed; "
                        "backup is at %s", rollback_exc, backup_dir,
                    )
                    rollback_note = f"rollback ALSO failed: {rollback_exc}"
                result = {
                    "status": "failed",
                    "reason": f"artifact swap or registry reload failed: {exc}",
                    "backup_dir": str(backup_dir),
                    "rollback": rollback_note,
                    "log_path": log_path,
                }
                _write_status(**result)
                return result

            result = {
                "status": "swapped", "comparison": comparison,
                "backup_dir": str(backup_dir), "log_path": log_path,
            }
            logger.info("Retraining improved (or matched) the deployed model - swapped in. %s", comparison)
        else:
            result = {
                "status": "rejected", "comparison": comparison, "log_path": log_path,
                "reason": "new model did not beat the currently deployed one - live artifacts untouched",
            }
            logger.warning("Retraining did NOT improve on the deployed model - keeping current artifacts. %s", comparison)

        _write_status(**result)
        return result
    except Exception as exc:
        # Catch-all so a truly unexpected failure (e.g. the training venv
        # is missing, or a metrics file is corrupt JSON) never leaves
        # retrain_status.json stuck on "running" forever - every path out
        # of this function writes a terminal status.
        result = {"status": "failed", "reason": f"unexpected error during retraining: {exc}"}
        _write_status(**result)
        logger.exception("Retraining failed with an unexpected error.")
        return result
    finally:
        await supabase_service.release_lock(RETRAIN_LOCK_NAME, triggered_by)
