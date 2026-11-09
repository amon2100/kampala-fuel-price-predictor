from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(
    r"C:\Users\CORE !5\Downloads\kampala-fuel-price-predictor"
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INCOMING_DIR = PROJECT_ROOT / "data" / "incoming"

REQUIRED_COLUMNS = [
    "month",
    "petrol_shs_per_litre",
    "diesel_shs_per_litre"
]


def get_latest_month(file_path):
    df = pd.read_csv(file_path)

    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        raise ValueError("Invalid fuel dataset structure.")

    dates = pd.to_datetime(df["month"], errors="coerce")

    if dates.isna().any():
        raise ValueError("Dataset contains invalid dates.")

    return dates.dt.to_period("M").max()


def monitor_releases():
    historical_file = (
        RAW_DIR / "uganda-fuel-prices-ubos-2026-09.csv"
    )

    if not historical_file.exists():
        print("Historical dataset not found.")
        return

    stored_month = get_latest_month(historical_file)

    print("LATEST STORED MONTH:", stored_month)

    incoming_files = list(INCOMING_DIR.glob("*.csv"))

    if not incoming_files:
        print("\nNo incoming files available.")
        return

    for file in incoming_files:

        try:
            incoming_month = get_latest_month(file)

            print("\nFILE:", file.name)
            print("LATEST MONTH:", incoming_month)

            if incoming_month > stored_month:
                print("STATUS: NEW RELEASE DETECTED")

            elif incoming_month == stored_month:
                print("STATUS: NO NEW MONTH DETECTED")

            else:
                print("STATUS: OLDER DATA RECEIVED")

        except Exception as error:
            print("STATUS: REQUIRES REVIEW")
            print("REASON:", error)


if __name__ == "__main__":
    monitor_releases()
