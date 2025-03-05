import argparse
import json
import pathlib

import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument(
    "--file_name",
    default="ND300",
    help="Interglad file found in data/raw/inter .",
    type=str,
)

parser.add_argument(
    "--property_name",
    default="RefractiveIndex",
    help="Property name.",
    type=str,
)


args = parser.parse_args()

RAW_DATA_PATH = pathlib.Path("data/raw/inter")

OUTPUT_PATH = pathlib.Path("data/processed")
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

file_name = args.file_name + ".pkl"
property_name = args.property_name

json_files = pathlib.Path("json")

with open(json_files / "compounds.json") as f:
    desired_compounds = json.load(f)
    desired_compounds = desired_compounds["desired_compounds"]


df_inter = pd.read_pickle(RAW_DATA_PATH / file_name)

df_inter = df_inter.drop(
    ["GLASS_ID", "YEAR", "PATENT_ID", "PATENT", "DSOURCE_DESC_EN", "AUTHOR"],
    axis=1,
)

df_inter.rename(columns={"PROP_FIG_SI": property_name}, inplace=True)

columns_to_remove = []

for coluna in df_inter.columns:
    if coluna not in desired_compounds + [property_name]:
        columns_to_remove.append(coluna)

processed_df = df_inter.drop(columns_to_remove, axis=1)

processed_df = processed_df.fillna(0)

row_sums = processed_df.drop(columns=["RefractiveIndex"]).sum(axis=1)

processed_df = processed_df[(row_sums >= 99) & (row_sums <= 101)]

file_name = f"processed_{args.file_name}-{args.property_name}.csv"

processed_df.to_csv(
    OUTPUT_PATH / file_name,
    index=False,
)
