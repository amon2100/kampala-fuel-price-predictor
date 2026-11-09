from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(
    r"C:\Users\CORE !5\Downloads\kampala-fuel-price-predictor"
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INCOMING_DIR = PROJECT_ROOT / "data" / "incoming"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HISTORICAL_FILE = (
    RAW_DIR / "uganda-fuel-prices-ubos-2026-09.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR / "uganda_fuel_prices_master.csv"
)

REQUIRED_COLUMNS = [
    "month",
    "petrol_shs_per_litre",
    "diesel_shs_per_litre"
]


def load_and_validate(file_path):
    df = pd.read_csv(file_path)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()

    df["month"] = pd.to_datetime(
        df["month"].astype(str),
        errors="coerce"
    )

    df["month"] = (
        df["month"].dt.to_period("M").dt.to_timestamp()
    )

    for col in REQUIRED_COLUMNS[1:]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    if df["month"].isna().any():
        raise ValueError("Invalid or missing month values.")

    if df[REQUIRED_COLUMNS[1:]].isna().any().any():
        raise ValueError("Missing or non-numeric fuel prices.")

    if (df[REQUIRED_COLUMNS[1:]] <= 0).any().any():
        raise ValueError("Zero or negative fuel prices found.")

    if df["month"].duplicated().any():
        raise ValueError("Duplicate months found.")

    return df.sort_values("month").reset_index(drop=True)


def update_master_dataset():
    if not HISTORICAL_FILE.exists():
        print("Historical UBOS dataset not found.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    master = load_and_validate(HISTORICAL_FILE)

    incoming_files = sorted(INCOMING_DIR.glob("*.csv"))

    if not incoming_files:
        print("No incoming files available.")
        print("Latest stored month:",
              master["month"].max().strftime("%Y-%m"))
        print("No changes made.")
        return

    for file in incoming_files:
        print(f"\nPROCESSING: {file.name}")
        print("-" * 50)

        try:
            incoming = load_and_validate(file)

            existing_months = set(master["month"])
            incoming_months = set(incoming["month"])

            overlap = existing_months.intersection(
                incoming_months
            )

            revisions = []

            for month in overlap:
                old_row = master.loc[
                    master["month"] == month
                ].iloc[0]

                new_row = incoming.loc[
                    incoming["month"] == month
                ].iloc[0]

                for col in REQUIRED_COLUMNS[1:]:
                    if old_row[col] != new_row[col]:
                        revisions.append(
                            (month.strftime("%Y-%m"), col,
                             old_row[col], new_row[col])
                        )

            if revisions:
                print("REVIEW REQUIRED: Historical revisions found.")

                for month, column, old, new in revisions:
                    print(
                        f"{month} | {column}: "
                        f"{old} -> {new}"
                    )

                print("File not merged.")
                continue

            new_records = incoming[
                ~incoming["month"].isin(existing_months)
            ]

            if new_records.empty:
                print("No new months detected.")
                continue

            master = pd.concat(
                [master, new_records],
                ignore_index=True
            )

            master = master.sort_values(
                "month"
            ).reset_index(drop=True)

            master.to_csv(OUTPUT_FILE, index=False)

            print("NEW MONTHS ADDED:",
                  len(new_records))

            print("LATEST MONTH:",
                  master["month"].max().strftime("%Y-%m"))

            print("UPDATED MASTER SAVED:")
            print(OUTPUT_FILE)

        except Exception as error:
            print("UPDATE REJECTED:", error)


if __name__ == "__main__":
    update_master_dataset()
