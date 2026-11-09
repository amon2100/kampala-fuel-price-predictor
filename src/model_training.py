from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(
    r"C:\Users\CORE !5\Downloads\kampala-fuel-price-predictor"
)

RAW_FILE = (
    PROJECT_ROOT / "data" / "raw"
    / "uganda-fuel-prices-ubos-2026-09.csv"
)

MASTER_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "uganda_fuel_prices_master.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
CANDIDATE_DIR = MODEL_DIR / "candidates"

FUEL_COLUMNS = {
    "petrol": "petrol_shs_per_litre",
    "diesel": "diesel_shs_per_litre"
}


def load_fuel_data():
    source = MASTER_FILE if MASTER_FILE.exists() else RAW_FILE

    if not source.exists():
        raise FileNotFoundError("No historical fuel dataset found.")

    df = pd.read_csv(source)

    required = ["month", *FUEL_COLUMNS.values()]

    if not all(col in df.columns for col in required):
        raise ValueError("Historical dataset has missing columns.")

    df["month"] = pd.to_datetime(df["month"], errors="coerce")

    for col in FUEL_COLUMNS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["month"].isna().any():
        raise ValueError("Invalid dates found.")

    if df["month"].duplicated().any():
        raise ValueError("Duplicate months found.")

    df = df.sort_values("month").reset_index(drop=True)

    return df


def build_training_data(df, fuel_column):
    records = []

    for i in range(2, len(df)):

        previous = df.iloc[i - 2]
        current = df.iloc[i - 1]
        target = df.iloc[i]

        # Only use three consecutive calendar months.
        consecutive = (
            current["month"] - previous["month"]
        ).days <= 32 and (
            target["month"].to_period("M")
            - current["month"].to_period("M")
        ).n == 1 and (
            current["month"].to_period("M")
            - previous["month"].to_period("M")
        ).n == 1

        if not consecutive:
            continue

        prices = [
            previous[fuel_column],
            current[fuel_column],
            target[fuel_column]
        ]

        if any(pd.isna(value) or value <= 0 for value in prices):
            continue

        records.append({
            "previous_month": previous["month"],
            "current_month": current["month"],
            "target_month": target["month"],
            "current_price": current[fuel_column],
            "previous_price": previous[fuel_column],
            "target": target[fuel_column]
        })

    result = pd.DataFrame(records)

    if len(result) < 30:
        raise ValueError(
            f"Insufficient training observations: {len(result)}"
        )

    return result


def evaluate_candidate(training_data):
    X = training_data[
        ["current_price", "previous_price"]
    ]

    y = training_data["target"]

    splitter = TimeSeriesSplit(
        n_splits=4,
        test_size=6
    )

    candidate_errors = []
    baseline_errors = []

    for train_index, test_index in splitter.split(X):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        baseline_predictions = X_test["current_price"]

        candidate_errors.append(
            mean_absolute_error(y_test, predictions)
        )

        baseline_errors.append(
            mean_absolute_error(y_test, baseline_predictions)
        )

    return {
        "candidate_mae": np.mean(candidate_errors),
        "baseline_mae": np.mean(baseline_errors)
    }


def train_candidate_models():
    df = load_fuel_data()

    print("LATEST DATA MONTH:", df["month"].max().strftime("%Y-%m"))
    print("TOTAL HISTORICAL ROWS:", len(df))

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    for fuel, column in FUEL_COLUMNS.items():

        print(f"\n{fuel.upper()} MODEL")
        print("-" * 45)

        training_data = build_training_data(df, column)

        X = training_data[
            ["current_price", "previous_price"]
        ]

        y = training_data["target"]

        metrics = evaluate_candidate(training_data)

        candidate = LinearRegression()
        candidate.fit(X, y)

        candidate_path = (
            CANDIDATE_DIR / f"{fuel}_candidate_model.pkl"
        )

        joblib.dump(candidate, candidate_path)

        print("Training observations:", len(training_data))
        print("Rolling CV candidate MAE:",
              round(metrics["candidate_mae"], 2))

        print("Rolling CV baseline MAE:",
              round(metrics["baseline_mae"], 2))

        if metrics["candidate_mae"] <= metrics["baseline_mae"]:
            print("BASELINE CHECK: PASSED")
        else:
            print("BASELINE CHECK: NOT PASSED")

        print("Candidate saved:", candidate_path)

    print("\nCANDIDATE TRAINING COMPLETED")
    print("Production models were not modified.")


if __name__ == "__main__":
    train_candidate_models()
