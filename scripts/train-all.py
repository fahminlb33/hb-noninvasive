import re
import glob

import numpy as np
import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, r_regression
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

def derive_signal_features_stats(ds: list[dict]):
    derive_ds = []

    # process each row
    for proc in ds:
        baseline = np.mean(proc["signal"][:2000])
        segments = np.split(proc["signal"][2000:], 30)
        corrected_segments = np.array([segment - baseline for segment in segments]).ravel()

        # calculate signal statistics
        derive_ds.append({
            "patient_name": proc["patient_name"].lower().strip(),
            "hb": proc["hb"],
            # "signal_corrected": corrected_segments
            
            **{f"avg_{k}": v for k, v in zip(range(len(corrected_segments)), corrected_segments)}
        })

    return pd.DataFrame(derive_ds)

def cross_val_model(path: str):
    # gold standard
    df_gold = pd.read_csv("../data/gold-09-nov.csv").drop(columns=["phone"])
    df_gold["name"] = df_gold["name"].str.lower().str.strip()
    df_gold.head()

    # device measurements
    dataset = []
    with open(path, "r") as f:
        for line in f:
            line = f.readline().split("|")
            if len(line) < 10:
                continue

            signal_trunc = np.array([float(x) for x in line[6:]]) # / 65535.0 # 16 bit ADC

            dataset.append({
                "num": int(line[0]),
                "patient_name": line[1].lower().strip(),
                "hb": float(line[5]),
                "signal": signal_trunc,
                "signal_ln_trunc": np.around(np.log(signal_trunc)),
            })

    # derive features
    df_signal = derive_signal_features_stats(dataset)
    df_all = df_signal.merge(df_gold, left_on="patient_name", right_on="name")

    dd = df_all.drop(columns=["patient_name", "name", "hb"], errors="ignore")
    r_hb = dd.corr()[["hb_gold"]]

    cols_corr = r_hb.sort_values("hb_gold",ascending=False).index.tolist()
    cols = [
        *cols_corr[1:4],
        *cols_corr[-3:],
    ]
    
    # get X, y
    X = df_all.drop(columns=["patient_name", "name", "hb", "hb_gold"], errors="ignore")[cols]
    y = df_all["hb_gold"]

    scoring = ["r2", "neg_mean_absolute_error", "neg_mean_squared_error", "neg_root_mean_squared_error"]

    scores = cross_validate(DecisionTreeRegressor(), X, y, scoring=scoring)
    scores_df = pd.DataFrame(scores)
    print(scores_df.mean())

def main():
    for file in glob.glob("../data/*.csv"):
        if "gold" in file:
            continue

        if "16 Nov" in file:
            continue

        print(file)
        cross_val_model(file)
        print("\n\n")

if __name__ == "__main__":
    main()
