from pathlib import Path

import pandas as pd


def main():
    outputs_dir = Path.cwd().joinpath("Outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data=[[2, 2], [3, 4]])
    df.to_csv(outputs_dir.joinpath("df.csv"))
