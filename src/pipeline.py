
from pathlib import Path
import subprocess
import sys
import hashlib
from datetime import datetime


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MASTER_FILE = (
    PROCESSED_DIR / "uganda_fuel_prices_master.csv"
)

LOG_FILE = OUTPUT_DIR / "pipeline_runs.log"


# ============================================================
# 2. FILE CHANGE DETECTION
# ============================================================

def get_file_hash(file_path):

    if not file_path.exists():
        return None

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(8192), b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 3. LOGGING
# ============================================================

def write_log(message):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    log_message = f"[{timestamp}] {message}"

    print(log_message)

    OUTPUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    with open(
        LOG_FILE, "a", encoding="utf-8"
    ) as file:

        file.write(log_message + "\n")


# ============================================================
# 4. EXECUTE A PROJECT MODULE
# ============================================================

def run_step(script_name):

    script_path = SRC_DIR / script_name

    if not script_path.exists():

        raise FileNotFoundError(
            f"Pipeline module not found: {script_path}"
        )

    write_log(f"STARTING: {script_name}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:

        write_log(
            f"FAILED: {script_name}"
        )

        raise RuntimeError(
            f"Pipeline stopped because {script_name} failed."
        )

    write_log(f"COMPLETED: {script_name}")


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def run_pipeline():

    write_log("=" * 60)
    write_log("UGANDAN FUEL PREDICTION PIPELINE STARTED")
    write_log("=" * 60)

    # Record the master dataset before updating.
    previous_hash = get_file_hash(MASTER_FILE)

    # STEP 1: Check for incoming data.
    run_step("release_monitor.py")

    # STEP 2: Validate and update the master dataset.
    run_step("data_update.py")

    # Check whether the master dataset changed.
    current_hash = get_file_hash(MASTER_FILE)

    if current_hash == previous_hash:

        write_log(
            "No new master dataset changes detected."
        )

        write_log(
            "Candidate training skipped."
        )

        write_log(
            "Existing production models remain unchanged."
        )

    else:

        write_log(
            "New or updated master dataset detected."
        )

        # STEP 3: Train candidate models.
        run_step("model_training.py")

        # STEP 4: Evaluate promotion decisions.
        run_step("model_promotion.py")

    write_log("=" * 60)
    write_log("PIPELINE EXECUTION COMPLETED")
    write_log("=" * 60)


# ============================================================
# 6. RUN CONTROLLER
# ============================================================

if __name__ == "__main__":

    try:

        run_pipeline()

    except Exception as error:

        write_log(
            f"PIPELINE ERROR: {error}"
        )

        raise
