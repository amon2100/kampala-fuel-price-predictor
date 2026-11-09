
from pathlib import Path
from datetime import datetime
import pandas as pd
import joblib


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uganda_fuel_prices_master.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_FILE = OUTPUT_DIR / "latest_fuel_forecast.csv"


# ============================================================
# FORECAST CONFIGURATION
# ============================================================

FUEL_CONFIG = {
    "Petrol": {
        "price_column": "petrol_shs_per_litre",
        "model_file": "petrol_momentum_model.pkl"
    },
    "Diesel": {
        "price_column": "diesel_shs_per_litre",
        "model_file": "diesel_momentum_model.pkl"
    }
}


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_master_dataset():

    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Master dataset not found: {MASTER_FILE}"
        )

    df = pd.read_csv(MASTER_FILE)

    required_columns = [
        "month",
        "petrol_shs_per_litre",
        "diesel_shs_per_litre"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing master dataset columns: {missing}"
        )

    df["month"] = pd.to_datetime(
        df["month"],
        errors="coerce"
    ).dt.to_period("M")

    if df["month"].isna().any():
        raise ValueError("Invalid dates in master dataset.")

    df = df.sort_values("month").reset_index(drop=True)

    if df["month"].duplicated().any():
        raise ValueError("Duplicate months in master dataset.")

    return df


# ============================================================
# LOAD APPROVED PRODUCTION MODELS
# ============================================================

def load_production_model(model_file):

    model_path = MODEL_DIR / model_file

    if not model_path.exists():
        raise FileNotFoundError(
            f"Production model not found: {model_path}"
        )

    return joblib.load(model_path)


# ============================================================
# GENERATE NEXT-MONTH FORECAST
# ============================================================

def generate_forecast(df, fuel, config):

    price_column = config["price_column"]

    fuel_data = df[
        ["month", price_column]
    ].dropna().copy()

    if len(fuel_data) < 2:
        raise ValueError(
            f"Insufficient historical data for {fuel}."
        )

    latest = fuel_data.iloc[-1]
    previous = fuel_data.iloc[-2]

    # Ensure the two observations are consecutive months.
    if latest["month"].ordinal - previous["month"].ordinal != 1:
        raise ValueError(
            f"Latest two observations for {fuel} "
            "are not consecutive months."
        )

    latest_price = float(latest[price_column])
    previous_price = float(previous[price_column])

    if latest_price <= 0 or previous_price <= 0:
        raise ValueError(
            f"Invalid historical prices for {fuel}."
        )

    model = load_production_model(
        config["model_file"]
    )

    # Preserve the exact production model feature order.
    feature_names = list(model.feature_names_in_)

    feature_values = {
        "current_price": latest_price,
        "previous_price": previous_price
    }

    if set(feature_names) != set(feature_values):
        raise ValueError(
            f"Unexpected model features for {fuel}: "
            f"{feature_names}"
        )

    X_latest = pd.DataFrame(
        [[feature_values[name] for name in feature_names]],
        columns=feature_names
    )

    predicted_price = float(
        model.predict(X_latest)[0]
    )

    if predicted_price <= 0:
        raise ValueError(
            f"Model generated an invalid forecast for {fuel}."
        )

    predicted_price = round(predicted_price)

    price_change = predicted_price - latest_price

    percentage_change = (
        price_change / latest_price
    ) * 100

    forecast_month = latest["month"] + 1

    return {
        "Fuel": fuel,
        "Latest_Actual_Month": str(latest["month"]),
        "Forecast_Month": str(forecast_month),
        "Latest_Actual_Price": round(latest_price),
        "Predicted_Price": predicted_price,
        "Price_Change": round(price_change),
        "Percentage_Change": round(
            percentage_change, 2
        ),
        "Model": "Momentum Linear Regression",
        "Generated_At": datetime.now().isoformat(
            timespec="seconds"
        )
    }


# ============================================================
# GENERATE ALL FUEL FORECASTS
# ============================================================

def run_forecast_generator():

    print("\nUGANDAN FUEL FORECAST GENERATOR")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_master_dataset()

    latest_month = df["month"].max()

    print(
        "LATEST ACTUAL MONTH:",
        latest_month
    )

    forecast_records = []

    for fuel, config in FUEL_CONFIG.items():

        print(f"\nGENERATING {fuel.upper()} FORECAST...")

        forecast = generate_forecast(
            df,
            fuel,
            config
        )

        forecast_records.append(forecast)

        print(
            "LATEST ACTUAL PRICE: UGX",
            f"{forecast['Latest_Actual_Price']:,}"
        )

        print(
            "PREDICTED PRICE: UGX",
            f"{forecast['Predicted_Price']:,}"
        )

        print(
            "FORECAST MONTH:",
            forecast["Forecast_Month"]
        )

    forecast_df = pd.DataFrame(
        forecast_records
    )

    forecast_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFORECAST GENERATION COMPLETED")
    print("=" * 60)

    print(forecast_df.to_string(index=False))

    print("\nSAVED TO:", OUTPUT_FILE)

    return forecast_df


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    run_forecast_generator()
