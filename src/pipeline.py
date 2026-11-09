
from pathlib import Path
import subprocess
import sys
import hashlib
import csv
from datetime import datetime


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

MASTER_FILE = (
    PROCESSED_DIR / "uganda_fuel_prices_master.csv"
)

FORECAST_FILE = (
    OUTPUT_DIR / "latest_fuel_forecast.csv"
)

LOG_FILE = OUTPUT_DIR / "pipeline_runs.log"

PRODUCTION_MODELS = [
    MODEL_DIR / "petrol_momentum_model.pkl",
    MODEL_DIR / "diesel_momentum_model.pkl"
]


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
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.stdout:

        print(result.stdout)

        for line in result.stdout.splitlines():
            write_log(f"[{script_name}] {line}")

    if result.stderr:

        print(result.stderr)

        for line in result.stderr.splitlines():
            write_log(f"[{script_name} STDERR] {line}")

    if result.returncode != 0:

        write_log(f"FAILED: {script_name}")

        raise RuntimeError(
            f"Pipeline stopped because {script_name} failed. "
            f"Exit code: {result.returncode}"
        )

    write_log(f"COMPLETED: {script_name}")


# ============================================================
# 5. VERIFY PRODUCTION MODELS
# ============================================================

def verify_production_models():

    missing_models = [
        model.name
        for model in PRODUCTION_MODELS
        if not model.exists()
    ]

    if missing_models:

        raise FileNotFoundError(
            "Missing production models: "
            + ", ".join(missing_models)
        )

    write_log(
        "Production petrol and diesel models verified."
    )


# ============================================================
# 6. CHECK FORECAST DATE
# ============================================================

def inspect_forecast_date():

    if not FORECAST_FILE.exists():

        raise FileNotFoundError(
            "Forecast file was not generated."
        )

    with open(
        FORECAST_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    if not rows:

        raise ValueError(
            "Forecast output contains no records."
        )

    current_month = datetime.now().strftime("%Y-%m")

    forecast_months = sorted({
        row["Forecast_Month"]
        for row in rows
        if row.get("Forecast_Month")
    })

    if not forecast_months:

        raise ValueError(
            "Forecast output has no valid Forecast_Month."
        )

    for forecast_month in forecast_months:

        if forecast_month < current_month:

            write_log(
                f"WARNING: {forecast_month} is earlier than "
                f"the current month {current_month}. "
                "This is an estimate for an unobserved "
                "historical month, not a future forecast."
            )

        elif forecast_month == current_month:

            write_log(
                f"Forecast target {forecast_month} is the "
                "current calendar month."
            )

        else:

            write_log(
                f"Future forecast target verified: "
                f"{forecast_month}"
            )

    write_log(
        f"Forecast output verified: {len(rows)} records."
    )


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def run_pipeline():

    OUTPUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    write_log("=" * 60)
    write_log("UGANDAN FUEL PREDICTION PIPELINE STARTED")
    write_log("=" * 60)

    # Capture dataset state before updating.
    previous_hash = get_file_hash(MASTER_FILE)

    # STEP 1: Monitor for incoming data.
    run_step("release_monitor.py")

    # STEP 2: Validate and update the master dataset.
    run_step("data_update.py")

    # STEP 3: Detect dataset changes.
    current_hash = get_file_hash(MASTER_FILE)

    if current_hash is None:

        raise FileNotFoundError(
            "Master dataset is missing after data update."
        )

    if current_hash == previous_hash:

        write_log(
            "No new master dataset changes detected."
        )

        write_log(
            "Candidate model training skipped."
        )

        write_log(
            "Existing production models remain unchanged."
        )

    else:

        write_log(
            "New or updated master dataset detected."
        )

        # STEP 4: Train candidate models.
        run_step("model_training.py")

        # STEP 5: Evaluate candidate model promotion.
        run_step("model_promotion.py")

        write_log(
            "Model promotion evaluation completed. "
            "Production models remain protected unless "
            "promotion is explicitly approved."
        )

    # STEP 6: Verify production models.
    verify_production_models()

    # STEP 7: Generate forecasts using production models.
    run_step("forecast_generator.py")

    # STEP 8: Verify forecast output and target dates.
    inspect_forecast_date()

    write_log("=" * 60)
    write_log("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    write_log("=" * 60)


# ============================================================
# 8. RUN CONTROLLER
# ============================================================

if __name__ == "__main__":

    try:

        run_pipeline()

    except Exception as error:

        write_log(
            f"PIPELINE ERROR: {error}"
        )

        raise
