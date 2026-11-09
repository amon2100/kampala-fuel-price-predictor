
from pathlib import Path
import os
import pandas as pd
import numpy as np


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INCOMING_DIR = PROJECT_ROOT / "data" / "incoming"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HISTORICAL_FILE = (
    RAW_DIR / "uganda-fuel-prices-ubos-2026-09.csv"
)

MASTER_FILE = (
    PROCESSED_DIR / "uganda_fuel_prices_master.csv"
)

REQUIRED_COLUMNS = [
    "month",
    "petrol_shs_per_litre",
    "diesel_shs_per_litre"
]

PRICE_COLUMNS = [
    "petrol_shs_per_litre",
    "diesel_shs_per_litre"
]


# ============================================================
# DATA VALIDATION
# ============================================================

def load_and_validate(file_path):
    """
    Load a fuel CSV, validate required fields and return
    a cleaned DataFrame.
    """

    df = pd.read_csv(file_path)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.copy()

    # Standardise month to the first day of each month.
    df["month"] = pd.to_datetime(
        df["month"],
        errors="coerce"
    ).dt.to_period("M").dt.to_timestamp()

    if df["month"].isna().any():
        raise ValueError("Invalid or missing month values.")

    # Validate fuel prices.
    for column in PRICE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if df[column].isna().any():
            raise ValueError(
                f"Missing or invalid prices in {column}."
            )

        if not np.isfinite(df[column]).all():
            raise ValueError(
                f"Non-finite values found in {column}."
            )

        if (df[column] <= 0).any():
            raise ValueError(
                f"Non-positive prices found in {column}."
            )

    # Reject duplicate months.
    if df["month"].duplicated().any():
        duplicates = df.loc[
            df["month"].duplicated(keep=False),
            "month"
        ].dt.strftime("%Y-%m").tolist()

        raise ValueError(
            f"Duplicate months detected: {duplicates}"
        )

    return df.sort_values("month").reset_index(drop=True)


# ============================================================
# HISTORICAL REVISION DETECTION
# ============================================================

def check_historical_revisions(master, incoming):
    """
    Compare overlapping months.

    Identical historical records are allowed.
    Changed petrol or diesel prices are rejected.
    """

    merged = master[
        ["month"] + PRICE_COLUMNS
    ].merge(
        incoming[["month"] + PRICE_COLUMNS],
        on="month",
        how="inner",
        suffixes=("_master", "_incoming")
    )

    for column in PRICE_COLUMNS:

        master_column = f"{column}_master"
        incoming_column = f"{column}_incoming"

        changed = ~np.isclose(
            merged[master_column],
            merged[incoming_column],
            rtol=0,
            atol=0.000001
        )

        if changed.any():
            affected_months = (
                merged.loc[changed, "month"]
                .dt.strftime("%Y-%m")
                .tolist()
            )

            raise ValueError(
                f"Historical revision detected in "
                f"{column}: {affected_months}"
            )


# ============================================================
# ATOMIC MASTER DATASET SAVING
# ============================================================

def save_master_atomically(master):
    """
    Save the master dataset through a temporary file.
    The existing master is replaced only after a successful write.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_file = MASTER_FILE.with_name(
        MASTER_FILE.stem + ".tmp.csv"
    )

    try:
        master.to_csv(
            temp_file,
            index=False
        )

        # Replace master only after successful temporary write.
        os.replace(
            temp_file,
            MASTER_FILE
        )

    finally:
        if temp_file.exists():
            temp_file.unlink()


# ============================================================
# MASTER INITIALISATION
# ============================================================

def initialize_master():
    """
    Create the initial processed master from the original
    historical UBOS dataset, but never overwrite an existing master.
    """

    if MASTER_FILE.exists():
        print("MASTER DATASET ALREADY EXISTS.")
        return

    if not HISTORICAL_FILE.exists():
        raise FileNotFoundError(
            f"Historical source not found: {HISTORICAL_FILE}"
        )

    historical = load_and_validate(
        HISTORICAL_FILE
    )

    save_master_atomically(historical)

    print("\nINITIAL MASTER DATASET CREATED")
    print("=" * 55)
    print(f"TOTAL RECORDS: {len(historical)}")
    print(
        f"FIRST MONTH: "
        f"{historical['month'].min():%Y-%m}"
    )
    print(
        f"LATEST MONTH: "
        f"{historical['month'].max():%Y-%m}"
    )
    print(f"SAVED TO: {MASTER_FILE}")


# ============================================================
# MASTER DATA UPDATE ENGINE
# ============================================================

def update_master_dataset():

    print("\nUGANDAN FUEL DATA UPDATE ENGINE")
    print("=" * 55)

    # --------------------------------------------------------
    # 1. Load the existing master as the source of truth.
    # --------------------------------------------------------

    if MASTER_FILE.exists():

        master = load_and_validate(
            MASTER_FILE
        )

        print(
            "MASTER DATASET LOADED:",
            len(master),
            "records"
        )

    else:

        print(
            "No existing master found. "
            "Initialising from historical UBOS data."
        )

        initialize_master()

        master = load_and_validate(
            MASTER_FILE
        )

    print(
        "LATEST MASTER MONTH:",
        master["month"].max().strftime("%Y-%m")
    )

    # --------------------------------------------------------
    # 2. Find incoming CSV files.
    # --------------------------------------------------------

    INCOMING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    incoming_files = sorted(
        INCOMING_DIR.glob("*.csv")
    )

    if not incoming_files:

        print("\nNo incoming files available.")
        print("MASTER DATASET REMAINS UNCHANGED.")

        return master

    print(
        f"\nINCOMING FILES FOUND: {len(incoming_files)}"
    )

    # --------------------------------------------------------
    # 3. Load and validate all incoming files.
    # --------------------------------------------------------

    incoming_frames = []

    for file_path in incoming_files:

        print(f"\nPROCESSING: {file_path.name}")

        incoming_df = load_and_validate(
            file_path
        )

        incoming_frames.append(incoming_df)

        print(
            f"VALIDATED RECORDS: {len(incoming_df)}"
        )

    incoming = pd.concat(
        incoming_frames,
        ignore_index=True
    )

    # Reject duplicate months across incoming files.
    if incoming["month"].duplicated().any():

        duplicates = (
            incoming.loc[
                incoming["month"].duplicated(keep=False),
                "month"
            ]
            .dt.strftime("%Y-%m")
            .tolist()
        )

        raise ValueError(
            "Duplicate months found across incoming files: "
            f"{duplicates}"
        )

    incoming = incoming.sort_values(
        "month"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 4. Detect historical revisions.
    # --------------------------------------------------------

    print("\nCHECKING HISTORICAL REVISIONS...")

    check_historical_revisions(
        master,
        incoming
    )

    print("NO HISTORICAL REVISIONS DETECTED.")

    # --------------------------------------------------------
    # 5. Identify genuinely new monthly records.
    # --------------------------------------------------------

    existing_months = set(
        master["month"]
    )

    new_records = incoming.loc[
        ~incoming["month"].isin(existing_months)
    ].copy()

    if new_records.empty:

        print("\nNO NEW MONTHLY RECORDS FOUND.")
        print("MASTER DATASET REMAINS UNCHANGED.")

        return master

    print(
        f"\nNEW RECORDS ACCEPTED: {len(new_records)}"
    )

    # --------------------------------------------------------
    # 6. Merge with existing master.
    # --------------------------------------------------------

    updated_master = pd.concat(
        [master, new_records],
        ignore_index=True,
        sort=False
    )

    updated_master = updated_master.sort_values(
        "month"
    ).reset_index(drop=True)

    # Validate final dataset before saving.
    if updated_master["month"].duplicated().any():
        raise ValueError(
            "Duplicate months detected in final master."
        )

    # --------------------------------------------------------
    # 7. Save atomically.
    # --------------------------------------------------------

    save_master_atomically(
        updated_master
    )

    print("\nMASTER DATASET UPDATED SUCCESSFULLY")
    print("=" * 55)
    print(f"PREVIOUS RECORDS: {len(master)}")
    print(f"NEW RECORDS: {len(new_records)}")
    print(f"TOTAL RECORDS: {len(updated_master)}")
    print(
        "LATEST MONTH:",
        updated_master["month"].max().strftime("%Y-%m")
    )
    print(f"SAVED TO: {MASTER_FILE}")

    return updated_master


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    update_master_dataset()
