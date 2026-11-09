import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(
    r"C:\Users\CORE !5\Downloads\kampala-fuel-price-predictor"
)

MODEL_DIR = PROJECT_ROOT / "models"
CANDIDATE_DIR = MODEL_DIR / "candidates"

RAW_FILE = (
    PROJECT_ROOT / "data" / "raw"
    / "uganda-fuel-prices-ubos-2026-09.csv"
)

MASTER_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "uganda_fuel_prices_master.csv"
)

FUEL_COLUMNS = {
    "petrol": "petrol_shs_per_litre",
    "diesel": "diesel_shs_per_litre"
}


def load_data():
    source = MASTER_FILE if MASTER_FILE.exists() else RAW_FILE
    df = pd.read_csv(source)

    df["month"] = pd.to_datetime(df["month"])

    return df.sort_values("month").reset_index(drop=True)


def build_data(df, column):
    records = []

    for i in range(2, len(df)):
        previous = df.iloc[i - 2]
        current = df.iloc[i - 1]
        target = df.iloc[i]

        if (
            current["month"].to_period("M")
            - previous["month"].to_period("M")
        ).n != 1:
            continue

        if (
            target["month"].to_period("M")
            - current["month"].to_period("M")
        ).n != 1:
            continue

        prices = [
            previous[column],
            current[column],
            target[column]
        ]

        if any(pd.isna(p) or p <= 0 for p in prices):
            continue

        records.append({
            "current_price": current[column],
            "previous_price": previous[column],
            "target": target[column]
        })

    return pd.DataFrame(records)


def compare_models():
    df = load_data()

    splitter = TimeSeriesSplit(
        n_splits=4,
        test_size=6
    )

    for fuel, column in FUEL_COLUMNS.items():

        print(f"\n{fuel.upper()} MODEL COMPARISON")
        print("=" * 55)

        data = build_data(df, column)

        X = data[["current_price", "previous_price"]]
        y = data["target"]

        production = joblib.load(
            MODEL_DIR / f"{fuel}_momentum_model.pkl"
        )

        candidate = joblib.load(
            CANDIDATE_DIR / f"{fuel}_candidate_model.pkl"
        )

        production_errors = []
        candidate_errors = []

        for train_index, test_index in splitter.split(X):

            X_test = X.iloc[test_index]
            y_test = y.iloc[test_index]

            production_predictions = production.predict(X_test)
            candidate_predictions = candidate.predict(X_test)

            production_errors.append(
                mean_absolute_error(
                    y_test, production_predictions
                )
            )

            candidate_errors.append(
                mean_absolute_error(
                    y_test, candidate_predictions
                )
            )

        production_mae = np.mean(production_errors)
        candidate_mae = np.mean(candidate_errors)

        improvement = (
            (production_mae - candidate_mae)
            / production_mae
        ) * 100

        print("Production MAE:", round(production_mae, 2))
        print("Candidate MAE:", round(candidate_mae, 2))
        print("Improvement (%):", round(improvement, 2))

        if candidate_mae < production_mae:
            print("RESULT: Candidate performs better.")
        else:
            print("RESULT: Keep production model.")


if __name__ == "__main__":
    compare_models()
