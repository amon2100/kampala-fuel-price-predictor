
from pathlib import Path
import importlib.util
import tempfile
import pandas as pd


# ============================================================
# LOAD THE REAL UPDATE ENGINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODULE_PATH = PROJECT_ROOT / "src" / "data_update.py"

spec = importlib.util.spec_from_file_location(
    "data_update",
    MODULE_PATH
)

data_update = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_update)


MASTER_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uganda_fuel_prices_master.csv"
)

original_master = pd.read_csv(MASTER_SOURCE)


# ============================================================
# TEST ENVIRONMENT
# ============================================================

def prepare_test_environment(temp_root):

    test_root = Path(temp_root)

    processed_dir = test_root / "processed"
    incoming_dir = test_root / "incoming"

    processed_dir.mkdir(parents=True)
    incoming_dir.mkdir(parents=True)

    master_file = (
        processed_dir / "uganda_fuel_prices_master.csv"
    )

    original_master.to_csv(
        master_file,
        index=False
    )

    # Redirect the engine to isolated test folders.
    data_update.PROCESSED_DIR = processed_dir
    data_update.INCOMING_DIR = incoming_dir
    data_update.MASTER_FILE = master_file

    return master_file, incoming_dir


def write_incoming(incoming_dir, filename, records):

    pd.DataFrame(records).to_csv(
        incoming_dir / filename,
        index=False
    )


def verify_master_unchanged(master_file, expected):

    actual = pd.read_csv(master_file)

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False
    )


# ============================================================
# TEST 1: ACCEPT A VALID NEW MONTH
# ============================================================

def test_valid_new_month():

    print("\nTEST 1: VALID NEW MONTH")

    with tempfile.TemporaryDirectory() as temp:

        master_file, incoming_dir = (
            prepare_test_environment(temp)
        )

        test_record = {
            "month": "2026-10",
            "petrol_shs_per_litre": 6700,
            "diesel_shs_per_litre": 6800
        }

        write_incoming(
            incoming_dir,
            "test_october.csv",
            [test_record]
        )

        result = data_update.update_master_dataset()

        assert len(result) == len(original_master) + 1

        assert (
            result["month"].max().strftime("%Y-%m")
            == "2026-10"
        )

        assert result.iloc[-1][
            "petrol_shs_per_litre"
        ] == 6700

        print("PASSED: New month successfully appended.")


# ============================================================
# TEST 2: REJECT HISTORICAL REVISION
# ============================================================

def test_historical_revision():

    print("\nTEST 2: HISTORICAL REVISION")

    with tempfile.TemporaryDirectory() as temp:

        master_file, incoming_dir = (
            prepare_test_environment(temp)
        )

        latest = original_master.iloc[-1]

        revised_record = {
            "month": latest["month"],
            "petrol_shs_per_litre":
                latest["petrol_shs_per_litre"] + 100,
            "diesel_shs_per_litre":
                latest["diesel_shs_per_litre"]
        }

        write_incoming(
            incoming_dir,
            "revision.csv",
            [revised_record]
        )

        try:

            data_update.update_master_dataset()

        except ValueError as error:

            assert "Historical revision detected" in str(error)

            verify_master_unchanged(
                master_file,
                original_master
            )

            print(
                "PASSED: Revision rejected; master unchanged."
            )

        else:

            raise AssertionError(
                "Historical revision was not rejected."
            )


# ============================================================
# TEST 3: REJECT INVALID PRICE
# ============================================================

def test_invalid_price():

    print("\nTEST 3: INVALID PRICE")

    with tempfile.TemporaryDirectory() as temp:

        master_file, incoming_dir = (
            prepare_test_environment(temp)
        )

        invalid_record = {
            "month": "2026-10",
            "petrol_shs_per_litre": -100,
            "diesel_shs_per_litre": 6800
        }

        write_incoming(
            incoming_dir,
            "invalid.csv",
            [invalid_record]
        )

        try:

            data_update.update_master_dataset()

        except ValueError:

            verify_master_unchanged(
                master_file,
                original_master
            )

            print(
                "PASSED: Invalid price rejected; "
                "master unchanged."
            )

        else:

            raise AssertionError(
                "Invalid price was not rejected."
            )


# ============================================================
# TEST 4: REJECT DUPLICATE MONTHS ACROSS FILES
# ============================================================

def test_duplicate_incoming_months():

    print("\nTEST 4: DUPLICATE INCOMING MONTHS")

    with tempfile.TemporaryDirectory() as temp:

        master_file, incoming_dir = (
            prepare_test_environment(temp)
        )

        record_a = {
            "month": "2026-10",
            "petrol_shs_per_litre": 6700,
            "diesel_shs_per_litre": 6800
        }

        record_b = {
            "month": "2026-10",
            "petrol_shs_per_litre": 6750,
            "diesel_shs_per_litre": 6850
        }

        write_incoming(
            incoming_dir,
            "file_a.csv",
            [record_a]
        )

        write_incoming(
            incoming_dir,
            "file_b.csv",
            [record_b]
        )

        try:

            data_update.update_master_dataset()

        except ValueError as error:

            assert "Duplicate months" in str(error)

            verify_master_unchanged(
                master_file,
                original_master
            )

            print(
                "PASSED: Duplicate incoming months rejected."
            )

        else:

            raise AssertionError(
                "Duplicate incoming months were accepted."
            )


# ============================================================
# TEST 5: ACCEPT IDENTICAL HISTORICAL RECORD
# ============================================================

def test_identical_historical_record():

    print("\nTEST 5: IDENTICAL HISTORICAL RECORD")

    with tempfile.TemporaryDirectory() as temp:

        master_file, incoming_dir = (
            prepare_test_environment(temp)
        )

        latest = original_master.iloc[-1]

        identical_record = {
            "month": latest["month"],
            "petrol_shs_per_litre":
                latest["petrol_shs_per_litre"],
            "diesel_shs_per_litre":
                latest["diesel_shs_per_litre"]
        }

        write_incoming(
            incoming_dir,
            "identical.csv",
            [identical_record]
        )

        result = data_update.update_master_dataset()

        assert len(result) == len(original_master)

        verify_master_unchanged(
            master_file,
            original_master
        )

        print(
            "PASSED: Identical historical record "
            "did not create a duplicate."
        )


# ============================================================
# TEST 6: TRANSACTION ROLLBACK
# ============================================================

def test_transaction_rollback():

    print("\nTEST 6: TRANSACTION ROLLBACK")

    with tempfile.TemporaryDirectory() as temp:

        master_file, incoming_dir = (
            prepare_test_environment(temp)
        )

        # First incoming file is valid.
        valid_record = {
            "month": "2026-10",
            "petrol_shs_per_litre": 6700,
            "diesel_shs_per_litre": 6800
        }

        # Second incoming file contains an invalid price.
        invalid_record = {
            "month": "2026-11",
            "petrol_shs_per_litre": -500,
            "diesel_shs_per_litre": 6900
        }

        # The valid file is deliberately processed first.
        write_incoming(
            incoming_dir,
            "a_valid.csv",
            [valid_record]
        )

        write_incoming(
            incoming_dir,
            "z_invalid.csv",
            [invalid_record]
        )

        try:

            data_update.update_master_dataset()

        except ValueError as error:

            assert "Non-positive prices" in str(error)

            # Confirm that the original master was not modified.
            verify_master_unchanged(
                master_file,
                original_master
            )

            print(
                "PASSED: Entire transaction rejected. "
                "Master dataset unchanged."
            )

        else:

            raise AssertionError(
                "Invalid batch was not rejected."
            )


# ============================================================
# RUN ALL TESTS
# ============================================================

def run_tests():

    print("\n" + "=" * 55)
    print("UGANDAN FUEL DATA INTEGRATION TESTS")
    print("=" * 55)

    tests = [
        test_valid_new_month,
        test_historical_revision,
        test_invalid_price,
        test_duplicate_incoming_months,
        test_identical_historical_record,
        test_transaction_rollback
    ]

    for test in tests:
        test()

    print("\n" + "=" * 55)
    print("ALL 6 INTEGRATION TESTS PASSED")
    print("=" * 55)


if __name__ == "__main__":
    run_tests()
