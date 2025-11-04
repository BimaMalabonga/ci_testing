# Python built-in packages
import pathlib

# Hard-coded project-related paths
PATHS_DIR = pathlib.Path(__file__)
ROOT_DIR = PATHS_DIR.parent
SOURCE_DIR = ROOT_DIR.joinpath("src")
TEST_DIR = ROOT_DIR.joinpath("tests")
TEST_UNIT_DIR = TEST_DIR.joinpath("unit")
TEST_SNAPSHOTS_DIR = TEST_DIR.joinpath("snapshots")
TEST_SNAPSHOTS_FILE = TEST_SNAPSHOTS_DIR.joinpath("test_snapshots.py")
MODEL_ENTRY_FILE = ROOT_DIR.joinpath("run.py")
TASKS_DIR = ROOT_DIR.joinpath("tasks.py")
README_DIR = ROOT_DIR.joinpath("README.md")
