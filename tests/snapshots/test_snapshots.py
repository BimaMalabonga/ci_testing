# Python built-in packages
import json
import os
import shutil
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Literal

# Third-party packages
import numpy as np
import pandas as pd
from detquantlib.data import DetDatabase
from dotenv import load_dotenv

# Internal modules
import paths
from src.main import main as main_model

# Constants
CASES_DIR = paths.TEST_SNAPSHOTS_DIR.joinpath("Cases")
EXPECTED_OUTPUTS_FOLDER = "ExpectedOutputs"
OUTPUTS_FOLDER_NAME = "Outputs"
INPUTS_FOLDER_NAME = "Inputs"
DATABASE_FOLDER_NAME = "_Database"
SETTINGS_FILENAME = "Settings.json"
DET_DB_SCHEMA = "SNAP"
ABSOLUTE_TOLERANCE = 1e-8
RELATIVE_TOLERANCE = 0

# Load environment variables from local file '.env' (if file exists)
load_dotenv()


def test_snapshots(run_type: Literal["compare", "update", "create"] = "compare"):
    """
    Main function to run snapshot tests. The function can fulfill one of three purposes:
        - Standard snapshot test run. The function then goes through the following steps:
            1. Find all existing test cases
            2. Run each test case
            3. Compare the actual model outputs with the expected outputs
        - Update the expected outputs of all test cases
        - Create a new test case, using the settings currently stored in the project's input folder

    Args:
        run_type: Determines the function's run type. Available options:
            - run_type="compare": Standard snapshot test run, comparing actual model outputs
                with expected outputs
            - run_type="update": Updates the expected outputs of all test cases
            - run_type="create": Creates a new test case, using the settings currently stored in
                the project's input folder

    Raises:
        ValueError: Raises an error if the value of the input argument 'run_type' is invalid.
        Exception: Raises an error if the model failed.
    """
    # Print line break
    print("")

    # Input validation
    accepted_run_types = ["compare", "update", "create"]
    if run_type not in accepted_run_types:
        accepted_run_types_str = ", ".join(f"'{item}'" for item in accepted_run_types)
        raise ValueError(
            f"Invalid input 'run_type'='{run_type}'. Accepted values: {accepted_run_types_str}."
        )

    # Find test cases
    if run_type == "create":
        cases = [create_new_case_folder()]
    else:
        cases = find_cases()

    for case_dir in cases:
        print("--------------------")
        print(f"Running test case '{case_dir.name}' ...")

        # Print case description, if it exists
        print_case_description(case_dir)

        # Change current working directory
        os.chdir(case_dir)

        # Define folder directories
        project_inputs_base_dir = paths.ROOT_DIR.joinpath(INPUTS_FOLDER_NAME)
        case_inputs_base_dir = case_dir.joinpath(INPUTS_FOLDER_NAME)
        expected_outputs_base_dir = case_dir.joinpath(EXPECTED_OUTPUTS_FOLDER)
        outputs_base_dir = case_dir.joinpath(OUTPUTS_FOLDER_NAME)

        # =======================================
        # Process inputs
        # =======================================

        # Dump mock data used for current test case in database
        table_names = dump_mock_data_in_db(case_dir)

        if run_type == "create":
            # Copy inputs folder to test case directory
            shutil.copytree(src=project_inputs_base_dir, dst=case_inputs_base_dir)

        # =======================================
        # Run model
        # =======================================

        try:
            # Run the model, muting everything that is printed within the model
            with open(os.devnull, "w") as fnull:
                with redirect_stdout(fnull), redirect_stderr(fnull):
                    main_model()
        except Exception as e:
            # If model fails, delete outputs folder and remove mock data from database before
            # raising the error
            remove_mock_data_from_db(table_names)
            if outputs_base_dir.is_dir():
                shutil.rmtree(outputs_base_dir)
            raise
        else:
            # If model succeeds, remove mock data from database
            remove_mock_data_from_db(table_names)

        # =======================================
        # Process outputs
        # =======================================

        if run_type == "create":
            # Store outputs in expected outputs folder
            copy_outputs(src=outputs_base_dir, dst=expected_outputs_base_dir)

        elif run_type == "update":
            # Store outputs in expected outputs folder
            copy_outputs(src=outputs_base_dir, dst=expected_outputs_base_dir)

        elif run_type == "compare":
            # Compare outputs with expected outputs
            compare_outputs(
                expected_outputs_base_dir=expected_outputs_base_dir,
                outputs_base_dir=outputs_base_dir,
            )

        # Remove outputs folder
        shutil.rmtree(outputs_base_dir)

        print(f"Test case '{case_dir.name}' finished successfully!")


def find_cases() -> list:
    """
    Finds all existing snapshot test cases.

    Returns:
        List of test case directories
    """
    cases = list()
    if CASES_DIR.is_dir():
        cases = [f for f in CASES_DIR.iterdir() if f.is_dir() and f.name.startswith("Case")]
    cases.sort()
    return cases


def create_new_case_folder() -> Path:
    """
    Creates the folder that will contain a new test case.

    Returns:
        Directory of the new test case
    """
    # Find all existing cases
    cases = find_cases()
    case_numbers = [int(c.name[4:]) for c in cases]

    # Determine new case number
    max_case_number = max(case_numbers) if len(case_numbers) > 0 else 0
    new_case_number = min([c for c in range(1, max_case_number + 2) if c not in case_numbers])

    # Create new case folder
    new_case = f"Case{new_case_number:02d}"
    new_case_dir = CASES_DIR.joinpath(new_case)
    new_case_dir.mkdir(parents=True, exist_ok=True)

    return new_case_dir


def print_case_description(case_dir: Path):
    """
    Prints the test case description, if it exists. Test case descriptions should be stored in
    the test case folder, in a file called case_description.txt.

    Args:
        case_dir: Test case folder directory
    """
    case_description_dir = case_dir.joinpath("case_description.txt")
    if case_description_dir.is_file():
        with open(case_description_dir, "r") as f:
            content = f.read()
            content = content.rstrip("\n")
            content = content.replace("\n", "\n> ")
            print("Case description:")
            print(f"> {content}")


def dump_mock_data_in_db(case_dir: Path) -> list[str]:
    """
    Dumps the mock input data used by the test case in the DET database.

    Args:
        case_dir: Test case folder directory

    Returns:
        List of table names containing the mock data

    Raises:
        Exception: Raises an error if database tables failed to be created.
    """
    # Initialize output
    table_names = list()

    # Read mock data settings
    database_dir = case_dir.joinpath(INPUTS_FOLDER_NAME, DATABASE_FOLDER_NAME)

    # If directory does not exist, exit function
    if not database_dir.is_dir():
        return table_names

    database_json_dir = database_dir.joinpath("_Database.json")
    with open(database_json_dir) as f:
        database_json = json.load(f)

    timestamp_unix = int(datetime.now().timestamp() * 1000)

    # Prepare mock data and settings. Note: We do not create the database tables in the same
    # step, to reduce the risk that tables are created and the code fails before they get deleted.
    mock_data = list()
    for d in database_json:
        # Define temporary table name
        table_name = f"{d['table_name']}_{timestamp_unix}"

        # Load mock data
        df = pd.read_parquet(database_dir.joinpath(d["filename"]))

        # Store table name in environment variable, so the model can access it
        os.environ[d["env_variable"]] = f"[{DET_DB_SCHEMA}].[{table_name}]"

        # Store mock data and settings
        mock_data.append(dict(table_name=table_name, data=df))

    db = DetDatabase()
    try:
        for md in mock_data:
            db.add_table(df=md["data"], table_name=md["table_name"], schema=DET_DB_SCHEMA)
            table_names.append(md["table_name"])
    except Exception as e:
        # If code fails, remove mock data from database before raising the error
        remove_mock_data_from_db(table_names)
        raise

    return table_names


def remove_mock_data_from_db(table_names: list[str]):
    """
    Removes the mock input data used by the test case from the DET database.

    Args:
        table_names: List of table names containing the mock data
    """
    if len(table_names) > 0:
        db = DetDatabase()
        for t in table_names:
            db.remove_table(table_name=t, schema=DET_DB_SCHEMA)


def copy_outputs(src: Path, dst: Path):
    """
    Copies output files and sub-folders from a source directory to a destination directory.

    Args:
        src: Source directory
        dst: Destination directory
    """
    # Hard-coded extensions that should not be copied
    extensions_to_ignore = [".html"]

    # Clear destination directory
    shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Loop over files and sub-folders located in source directory
    for root, _, filenames in os.walk(src):
        # Define root in destination directory
        dir_suffix = Path(root).relative_to(src)
        dst_root = dst.joinpath(dir_suffix)

        for f in filenames:
            # Define file path in source directory and destination directory
            src_file_dir = Path(root).joinpath(f)
            dst_file_dir = dst_root.joinpath(f)

            if src_file_dir.suffix not in extensions_to_ignore:
                # Create destination folder (only if extension is not ignored)
                dst_root.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(src=src_file_dir, dst=dst_file_dir)


def compare_outputs(expected_outputs_base_dir: Path, outputs_base_dir: Path):
    """
    Compare the model outputs with the expected outputs, and flags differences.

    Args:
        expected_outputs_base_dir: Expected outputs folder directory
        outputs_base_dir: Model outputs folder directory

    Raises:
        Exception: Raises an error if the file assertion fails
    """
    # Loop over files and sub-folders located the expected outputs folder
    for root, _, filenames in os.walk(expected_outputs_base_dir):
        for f in filenames:
            # Define path in expected outputs folder
            expected_output_file_dir = Path(root).joinpath(f)

            # Define path in model outputs folder
            dir_suffix = expected_output_file_dir.relative_to(expected_outputs_base_dir)
            output_file_dir = outputs_base_dir.joinpath(dir_suffix)

            print(f"Asserting output file: '{dir_suffix}'")

            # Assert output file
            try:
                assert_file(output_file_dir, expected_output_file_dir)
            except Exception as e:
                # Remove outputs folder, then raise the original exception
                shutil.rmtree(outputs_base_dir)
                raise e


def assert_file(output_file_dir: Path, expected_output_file_dir: Path):
    """
    Entry point function to assert that an output file is equal to the corresponding expected
    output file. Depending on the file extension, the procedure to open and assert the files
    will differ.

    Args:
        output_file_dir: Output file directory
        expected_output_file_dir: Expected output file directory

    Raises:
        ValueError: Raises an error if the file extension is not supported
    """
    extension = expected_output_file_dir.suffix
    if extension == ".csv":
        assert_csv(output_file_dir, expected_output_file_dir)
    elif extension == ".json":
        assert_json(output_file_dir, expected_output_file_dir)
    elif extension == ".npz":
        assert_npz(output_file_dir, expected_output_file_dir)
    elif extension == ".html":
        # We do not store html files, because they contain metadata values that change every
        # time (such as unique identifiers).
        pass
    else:
        raise ValueError(f"File extension '{extension}' is not supported.")


def assert_csv(output_file_dir: Path, expected_output_file_dir: Path):
    """
    Imports an output csv file and asserts that it is equal to the corresponding expected
    output csv file.

    Args:
        output_file_dir: Output file directory
        expected_output_file_dir: Expected output file directory
    """
    data = pd.read_csv(output_file_dir)
    expected_data = pd.read_csv(expected_output_file_dir)
    pd.testing.assert_frame_equal(
        data, expected_data, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE
    )


def assert_json(output_file_dir: Path, expected_output_file_dir: Path):
    """
    Imports an output json file and asserts that it is equal to the corresponding expected
    output json file.

    Args:
        output_file_dir: Output file directory
        expected_output_file_dir: Expected output file directory
    """
    with open(output_file_dir) as f:
        data = json.load(f)
    with open(expected_output_file_dir) as f:
        expected_data = json.load(f)
    assert data == expected_data


def assert_npz(output_file_dir: Path, expected_output_file_dir: Path):
    """
    Imports an output npz file and asserts that it is equal to the corresponding expected
    output npz file.

    Args:
        output_file_dir: Output file directory
        expected_output_file_dir: Expected output file directory

    Raises:
        KeyError: Raises an error when the keys of the npz archive files do not match.
    """
    data = np.load(output_file_dir)
    expected_data = np.load(expected_output_file_dir)
    if set(data.files) != set(expected_data.files):
        raise KeyError("The .npz archives have different keys.")
    else:
        # Compare all files in archives
        for name in expected_data.files:
            np.testing.assert_allclose(
                data[name], expected_data[name], atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE
            )
