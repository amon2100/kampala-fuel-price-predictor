
from pathlib import Path
import pandas as pd
import numpy as np

# Automatically locate the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

PRICE_COLUMNS = REQUIRED_COLUMNS[1:]


def load_and_validate(file_path):
    df = pd.read_csv(file_path)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Preserve additional columns.
    df = df.copy()

    df["month"] = pd.to_datetime(
        df["month"].astype(str),
        errors="coerce"
    )

    df["month"] = (
        df["month"].dt.to_period("M").dt.to_timestamp()
    )

    if df["month"].isna().any():
        raise ValueError("Invalid or missing month values.")

    for col in PRICE_COLUMNS:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    if df[PRICE_COLUMNS].isna().any().any():
        raise ValueError("Missing or non-numeric fuel prices.")

    if not np.isfinite(df[PRICE_COLUMNS].to_numpy()).all():
        raise ValueError("Infinite fuel prices found.")

    if (df[PRICE_COLUMNS] <= 0).any().any():
        raise ValueError("Zero or negative fuel prices found.")

    if df["month"].duplicated().any():
        raise ValueError("Duplicate months found.")

    return df.sort_values("month").reset_index(drop=True)


def check_historical_revisions(master, incoming):
    existing_months = set(master["month"])
    incoming_months = set(incoming["month"])

    overlap = existing_months.intersection(incoming_months)

    revisions = []

    for month in overlap:
        old_row = master.loc[
            master["month"] == month
        ].iloc[0]

        new_row = incoming.loc[
            incoming["month"] == month
        ].iloc[0]

        for col in PRICE_COLUMNS:
            if old_row[col] != new_row[col]:
                revisions.append(
                    (
                        month.strftime("%Y-%m"),
                        col,
                        old_row[col],
                        new_row[col]
                    )
                )

    return revisions


def save_master_atomically(master):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    master.to_csv(temporary_file, index=False)

    # Replace the master only after the temporary file is written.
    temporary_file.replace(OUTPUT_FILE)


def update_master_dataset():

    if not HISTORICAL_FILE.exists():
        print("Historical UBOS dataset not found.")
        return

    master = load_and_validate(HISTORICAL_FILE)

    incoming_files = sorted(
        INCOMING_DIR.glob("*.csv")
    ) if INCOMING_DIR.exists() else []

    if not incoming_files:
        print("No incoming files available.")
        print(
            "Latest stored month:",
            master["month"].max().strftime("%Y-%m")
        )
        print("No changes made.")
        return

    # Work on a temporary in-memory copy.
    # The real master file is not changed until all files pass.
    candidate = master.copy()

    incoming_months_seen = set()
    total_new_records = 0

    print("STARTING TRANSACTIONAL DATA UPDATE")
    print("=" * 55)

    try:
        for file in incoming_files:

            print(f"\nPROCESSING: {file.name}")
            print("-" * 50)

            incoming = load_and_validate(file)

            # Prevent different incoming files from supplying
            # conflicting or duplicate months.
            duplicate_months = (
                incoming_months_seen.intersection(
                    set(incoming["month"])
                )
            )

            if duplicate_months:
                months = sorted(
                    month.strftime("%Y-%m")
                    for month in duplicate_months
                )

                raise ValueError(
                    f"Months appear in multiple incoming files: {months}"
                )

            incoming_months_seen.update(incoming["month"])

            revisions = check_historical_revisions(
                candidate,
                incoming
            )

            if revisions:
                print("HISTORICAL REVISION DETECTED")

                for month, column, old, new in revisions:
                    print(
                        f"{month} | {column}: "
                        f"{old} -> {new}"
                    )

                raise ValueError(
                    "Historical revisions require manual review."
                )

            existing_months = set(candidate["month"])

            new_records = incoming.loc[
                ~incoming["month"].isin(existing_months)
            ].copy()

            if new_records.empty:
                print("No new months detected.")
                continue

            # Combine columns without discarding additional fields.
            candidate = pd.concat(
                [candidate, new_records],
                ignore_index=True,
                sort=False
            )

            candidate = candidate.sort_values(
                "month"
            ).reset_index(drop=True)

            total_new_records += len(new_records)

            print("NEW MONTHS FOUND:", len(new_records))

        # Final integrity checks before committing.
        if candidate["month"].duplicated().any():
            raise ValueError(
                "Duplicate months detected in final dataset."
            )

        if total_new_records == 0:
            print("\nNo new records to save.")
            print("Master dataset remains unchanged.")
            return

        # Commit only after every incoming file passes.
        save_master_atomically(candidate)

        print("\n" + "=" * 55)
        print("DATA UPDATE COMMITTED SUCCESSFULLY")
        print("=" * 55)

        print("NEW RECORDS ADDED:", total_new_records)

        print(
            "LATEST MONTH:",
            candidate["month"].max().strftime("%Y-%m")
        )

        print("MASTER DATASET SAVED:")
        print(OUTPUT_FILE)

        print("ALL INCOMING FILES PROCESSED SUCCESSFULLY.")

    except Exception as error:
        print("\n" + "=" * 55)
        print("TRANSACTION ABORTED")
        print("=" * 55)

        print("REASON:", error)
        print("No changes were committed to the master dataset.")


if __name__ == "__main__":
    update_master_dataset()
