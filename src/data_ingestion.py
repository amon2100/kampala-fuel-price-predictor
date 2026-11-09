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


def find_fuel_files():
    files = []

    for folder in [INCOMING_DIR, RAW_DIR]:
        if folder.exists():
            files.extend(folder.glob("*.csv"))

    return sorted(
        [
            file for file in files
            if "fuel" in file.name.lower()
            or "ubos" in file.name.lower()
        ],
        key=lambda file: file.stat().st_mtime,
        reverse=True
    )


def inspect_fuel_file(file_path):
    df = pd.read_csv(file_path)

    print("\\nFILE:", file_path.name)
    print("-" * 50)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        print("STATUS: INVALID")
        print("Missing columns:", missing_columns)
        return None

    df["month"] = pd.to_datetime(
        df["month"].astype(str),
        errors="coerce"
    )

    print("STATUS: VALID BASIC STRUCTURE")
    print("Rows:", len(df))
    print("Date range:", df["month"].min(), "to", df["month"].max())
    print("Duplicate months:", df["month"].duplicated().sum())

    print("\\nMissing values:")
    print(df[REQUIRED_COLUMNS].isnull().sum())

    print("\\nLatest records:")
    print(df.sort_values("month").tail(5).to_string(index=False))

    return df


def main():
    files = find_fuel_files()

    if not files:
        print("No fuel-price CSV files found.")
        return

    print("FUEL DATA FILES FOUND:", len(files))

    for file in files:
        inspect_fuel_file(file)


if __name__ == "__main__":
    main()
