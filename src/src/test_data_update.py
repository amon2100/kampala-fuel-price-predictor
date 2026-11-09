
from pathlib import Path
import pandas as pd
import tempfile
import shutil

# ============================================================
# TEST CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uganda_fuel_prices_master.csv"
)

TEST_COLUMNS = [
    "month",
    "petrol_shs_per_litre",
    "diesel_shs_per_litre"
]


def prepare_test_environment():

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            "Processed master dataset not found."
        )

    original = pd.read_csv(SOURCE_FILE)

    return original


def validate_test_data(df):

    missing = [
        col for col in TEST_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df.copy()

    df["month"] = pd.to_datetime(
        df["month"],
        errors="coerce"
    )

    for col in TEST_COLUMNS[1:]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if df[TEST_COLUMNS].isna().any().any():
        raise ValueError(
            "Missing or invalid required values."
        )

    if (df[TEST_COLUMNS[1:]] <= 0).any().any():
        raise ValueError(
            "Non-positive fuel prices."
        )

    if df["month"].duplicated().any():
        raise ValueError(
            "Duplicate months."
        )

    return df


def test_valid_new_month(original):

    print("\nTEST 1: VALID NEW MONTH")
    print("-" * 45)

    test_data = original.copy()

    new_record = {
        "month": "2026-10-01",
        "petrol_shs_per_litre": 6700,
        "diesel_shs_per_litre": 6800
    }

    test_data = pd.concat(
        [
            test_data,
            pd.DataFrame([new_record])
        ],
        ignore_index=True
    )

    result = validate_test_data(test_data)

    assert len(result) == len(original) + 1

    print("PASSED: Valid test record accepted.")


def test_invalid_price(original):

    print("\nTEST 2: INVALID PRICE")
    print("-" * 45)

    test_data = original.copy()

    invalid_record = {
        "month": "2026-10-01",
        "petrol_shs_per_litre": -100,
        "diesel_shs_per_litre": 6800
    }

    test_data = pd.concat(
        [
            test_data,
            pd.DataFrame([invalid_record])
        ],
        ignore_index=True
    )

    try:

        validate_test_data(test_data)

    except ValueError:

        print("PASSED: Invalid price rejected.")

        return

    raise AssertionError(
        "Invalid price was not rejected."
    )


def test_historical_revision(original):

    print("\nTEST 3: HISTORICAL REVISION")
    print("-" * 45)

    test_data = original.copy()

    test_data.loc[
        test_data["month"] == "2026-09-01",
        "petrol_shs_per_litre"
    ] = 7000

    try:

        check = test_data.merge(
            original[
                ["month", "petrol_shs_per_litre"]
            ],
            on="month",
            suffixes=("_new", "_old")
        )

        revisions = check[
            check["petrol_shs_per_litre_new"]
            != check["petrol_shs_per_litre_old"]
        ]

        if not revisions.empty:

            raise ValueError(
                "Historical revision detected."
            )

    except ValueError:

        print("PASSED: Historical revision detected.")

        return

    raise AssertionError(
        "Historical revision was not detected."
    )


def run_tests():

    print("=" * 55)
    print("UGANDAN FUEL DATA UPDATE - TEST SUITE")
    print("=" * 55)

    original = prepare_test_environment()

    print("SOURCE RECORDS:", len(original))

    test_valid_new_month(original)

    test_invalid_price(original)

    test_historical_revision(original)

    print("\n" + "=" * 55)
    print("ALL DATA UPDATE TESTS COMPLETED")
    print("=" * 55)


if __name__ == "__main__":

    run_tests()
