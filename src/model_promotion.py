
from pathlib import Path
from datetime import datetime
import shutil
import json

import pandas as pd
import numpy as np
import joblib

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\CORE !5\Downloads\kampala-fuel-price-predictor"
)

MODEL_DIR = PROJECT_ROOT / "models"
CANDIDATE_DIR = MODEL_DIR / "candidates"
BACKUP_DIR = MODEL_DIR / "backups"

RAW_FILE = (
    PROJECT_ROOT / "data" / "raw"
    / "uganda-fuel-prices-ubos-2026-09.csv"
)

MASTER_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "uganda_fuel_prices_master.csv"
)

HISTORY_FILE = (
    PROJECT_ROOT / "outputs"
    / "model_promotion_history.csv"
)

FUEL_COLUMNS = {
    "petrol": "petrol_shs_per_litre",
    "diesel": "diesel_shs_per_litre"
}

# SAFETY: Keep False until promotion logic is verified.
PROMOTE_APPROVED = False

# Candidate must beat persistence baseline by at least this amount.
MINIMUM_IMPROVEMENT_PERCENT = 0.0

N_SPLITS = 4
TEST_SIZE = 6


# ============================================================
# 2. LOAD HISTORICAL DATA
# ============================================================

def load_data():

    source = MASTER_FILE if MASTER_FILE.exists() else RAW_FILE

    if not source.exists():
        raise FileNotFoundError(
            f"Historical fuel dataset not found: {source}"
        )

    df = pd.read_csv(source)

    required_columns = [
        "month",
        "petrol_shs_per_litre",
        "diesel_shs_per_litre"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["month"] = pd.to_datetime(
        df["month"], errors="coerce"
    )

    df = df.dropna(subset=["month"])

    if df["month"].duplicated().any():
        raise ValueError(
            "Duplicate months found in historical dataset."
        )

    return df.sort_values("month").reset_index(drop=True)


# ============================================================
# 3. BUILD CONSECUTIVE-MONTH TRAINING OBSERVATIONS
# ============================================================

def build_data(df, column):

    records = []

    for i in range(2, len(df)):

        previous = df.iloc[i - 2]
        current = df.iloc[i - 1]
        target = df.iloc[i]

        previous_month = previous["month"].to_period("M")
        current_month = current["month"].to_period("M")
        target_month = target["month"].to_period("M")

        # Only use consecutive months.
        if (current_month - previous_month).n != 1:
            continue

        if (target_month - current_month).n != 1:
            continue

        prices = [
            previous[column],
            current[column],
            target[column]
        ]

        prices = pd.to_numeric(
            pd.Series(prices), errors="coerce"
        )

        if prices.isna().any() or (prices <= 0).any():
            continue

        records.append({
            "current_price": float(current[column]),
            "previous_price": float(previous[column]),
            "target": float(target[column])
        })

    return pd.DataFrame(records)


# ============================================================
# 4. CHRONOLOGICAL MODEL EVALUATION
# ============================================================

def evaluate_candidate(candidate, data):

    X = data[[
        "current_price",
        "previous_price"
    ]]

    y = data["target"]

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE
    )

    candidate_errors = []
    baseline_errors = []

    for train_index, test_index in splitter.split(X):

        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]

        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        # Clone the estimator so each fold starts unfitted.
        fold_model = clone(candidate)

        # Fit only on the training portion of this fold.
        fold_model.fit(X_train, y_train)

        predictions = fold_model.predict(X_test)

        # Persistence baseline:
        # next month's price = current month's price.
        baseline_predictions = X_test["current_price"]

        candidate_errors.append(
            mean_absolute_error(
                y_test, predictions
            )
        )

        baseline_errors.append(
            mean_absolute_error(
                y_test, baseline_predictions
            )
        )

    candidate_mae = float(np.mean(candidate_errors))
    baseline_mae = float(np.mean(baseline_errors))

    improvement = (
        (baseline_mae - candidate_mae)
        / baseline_mae
    ) * 100 if baseline_mae > 0 else 0.0

    return {
        "candidate_mae": candidate_mae,
        "baseline_mae": baseline_mae,
        "improvement_percent": improvement
    }


# ============================================================
# 5. CHECK WHETHER MODELS ARE ALREADY IDENTICAL
# ============================================================

def models_are_identical(production, candidate):

    same_coefficients = np.allclose(
        production.coef_,
        candidate.coef_
    )

    same_intercept = np.isclose(
        production.intercept_,
        candidate.intercept_
    )

    return same_coefficients and same_intercept


# ============================================================
# 6. RECORD PROMOTION DECISION
# ============================================================

def record_decision(record):

    HISTORY_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    new_record = pd.DataFrame([record])

    if HISTORY_FILE.exists():

        old_records = pd.read_csv(HISTORY_FILE)

        new_record = pd.concat(
            [old_records, new_record],
            ignore_index=True
        )

    new_record.to_csv(
        HISTORY_FILE, index=False
    )


# ============================================================
# 7. SAFELY PROMOTE AN APPROVED MODEL
# ============================================================

def promote_model(fuel, candidate):

    production_path = (
        MODEL_DIR / f"{fuel}_momentum_model.pkl"
    )

    BACKUP_DIR.mkdir(
        parents=True, exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_path = (
        BACKUP_DIR
        / f"{fuel}_production_backup_{timestamp}.pkl"
    )

    # Preserve the current production model.
    shutil.copy2(
        production_path,
        backup_path
    )

    # Save the candidate to a temporary file first.
    temporary_path = production_path.with_name(
        production_path.stem + "_new.pkl"
    )

    joblib.dump(
        candidate,
        temporary_path
    )

    # Replace production only after successful saving.
    temporary_path.replace(production_path)

    print("MODEL PROMOTED:", production_path)
    print("BACKUP CREATED:", backup_path)


# ============================================================
# 8. MAIN PROMOTION PROCESS
# ============================================================

def run_promotion():

    MODEL_DIR.mkdir(
        parents=True, exist_ok=True
    )

    df = load_data()

    print("\nUGANDAN FUEL MODEL PROMOTION")
    print("=" * 60)

    print(
        "Latest historical month:",
        df["month"].max().strftime("%Y-%m")
    )

    print("Historical rows:", len(df))

    if PROMOTE_APPROVED:
        print("MODE: LIVE PROMOTION ENABLED")
    else:
        print("MODE: DRY RUN - NO PRODUCTION CHANGES")

    for fuel, column in FUEL_COLUMNS.items():

        print(f"\n{fuel.upper()} MODEL")
        print("-" * 60)

        production_path = (
            MODEL_DIR / f"{fuel}_momentum_model.pkl"
        )

        candidate_path = (
            CANDIDATE_DIR / f"{fuel}_candidate_model.pkl"
        )

        if not production_path.exists():
            raise FileNotFoundError(
                f"Production model missing: {production_path}"
            )

        if not candidate_path.exists():
            raise FileNotFoundError(
                f"Candidate model missing: {candidate_path}"
            )

        production = joblib.load(production_path)
        candidate = joblib.load(candidate_path)

        data = build_data(df, column)

        if len(data) < N_SPLITS * TEST_SIZE + 2:
            raise ValueError(
                f"Not enough observations for {fuel} validation."
            )

        results = evaluate_candidate(
            candidate, data
        )

        print(
            "Candidate rolling MAE:",
            round(results["candidate_mae"], 2)
        )

        print(
            "Persistence baseline MAE:",
            round(results["baseline_mae"], 2)
        )

        print(
            "Improvement over baseline (%):",
            round(results["improvement_percent"], 2)
        )

        baseline_passed = (
            results["improvement_percent"]
            > MINIMUM_IMPROVEMENT_PERCENT
        )

        identical = models_are_identical(
            production, candidate
        )

        if not baseline_passed:

            decision = "REJECTED"
            reason = "Candidate did not beat persistence baseline."

        elif identical:

            decision = "UNCHANGED"
            reason = "Candidate is identical to production model."

        elif not PROMOTE_APPROVED:

            decision = "APPROVED - DRY RUN"
            reason = "Candidate passed validation; promotion disabled."

        else:

            promote_model(fuel, candidate)

            decision = "PROMOTED"
            reason = "Candidate passed validation and was promoted."

        print("Model identical to production:", identical)
        print("DECISION:", decision)
        print("REASON:", reason)

        record_decision({
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "fuel": fuel,
            "latest_data_month": df["month"].max().strftime("%Y-%m"),
            "observations": len(data),
            "candidate_mae": round(
                results["candidate_mae"], 2
            ),
            "baseline_mae": round(
                results["baseline_mae"], 2
            ),
            "improvement_percent": round(
                results["improvement_percent"], 2
            ),
            "identical_to_production": identical,
            "decision": decision,
            "reason": reason
        })

    print("\nPROMOTION PROCESS COMPLETED.")
    print("Promotion history:", HISTORY_FILE)


if __name__ == "__main__":
    run_promotion()
