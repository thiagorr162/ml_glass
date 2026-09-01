## Analyzing and cleaning the dataset.

import numpy as np
import pandas as pd

data = pd.read_csv("processed_ND300-RefractiveIndex.csv") 
RI = "RefractiveIndex" 
raw_rows = len(data)
oxide_columns = [column for column in data.columns if column != RI] #This saves the oxide columns (removes the RI).

data=data.apply(pd.to_numeric, errors="coerce") #Converts the data to numbers.


valid_rows= (
    data[RI].between(1.0,5.0)
    & data.notna().all(axis=1)
) #Separates the valid rows as the ones which have the correct RI interval (between 1 and 5) and are not NaN.

totals = data[oxide_columns].sum(axis=1) #Sums all of the concentrations for every row.
valid_rows=valid_rows & (totals>0)
data=data.loc[valid_rows].copy()
data[oxide_columns]= (data[oxide_columns].div(totals, axis=0).round(12)) #Normalizes each concentration so that they sum to 1.

unique = (
    data.groupby(oxide_columns, as_index=False)
    .agg(
        RI_mean=(RI, "mean"),
        RI_min=(RI, "min"),
        RI_max=(RI, "max"),
        replicate_count=(RI, "size")
    )
)

unique["RI_conflict"] = (
    unique["RI_max"] - unique["RI_min"] > 0.01
)

statistics = pd.Series({
      "raw_rows": raw_rows,
      "clean_rows": len(data),
      "rejected_rows": raw_rows - len(data),
      "unique_compositions": len(unique),
      "replicate_groups": (unique["replicate_count"] > 1).sum(),
      "conflicting_groups": unique["RI_conflict"].sum(),
      "RI_mean": data[RI].mean(),
      "RI_median": data[RI].median(),
      "RI_minimum": data[RI].min(),
      "RI_maximum": data[RI].max(),
      "zero_fraction": data[oxide_columns].eq(0).mean().mean(),
      "median_nonzero_oxides": data[oxide_columns].gt(0).sum(axis=1).median(),
  })

data.to_csv("cleaned_rows.csv", index=False)
unique.to_csv("cleaned_unique_compositions.csv", index=False)
statistics.to_csv("data_statistics.csv", header=["value"])
