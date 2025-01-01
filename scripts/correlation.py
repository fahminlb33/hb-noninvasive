import argparse

import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import pearsonr
from rich import print as print_pretty


def main(args):
    # open dataset
    ds_hb = xr.open_dataset(args.dataset_file)

    # get X and y
    X = ds_hb["signal"].to_numpy()
    y = ds_hb["hb"].to_numpy()

    # calculate pearson correlation
    r, p = pearsonr(X, y.reshape(-1, 1))

    # save into frame
    df_corr = pd.DataFrame(
        {
            "feat_num": range(X.shape[1]),
            "r": r,
            "r_abs": np.abs(r),
            "p_value": p,
        }
    )

    # sort and print
    df_corr = df_corr.sort_values("r_abs", ascending=False)
    print_pretty(df_corr.head())

    # save to file
    df_corr.to_csv(args.output_file, index=None)


if __name__ == "__main__":
    # create CLI parser
    parser = argparse.ArgumentParser()

    # add dataset and output paths
    parser.add_argument("dataset_file", type=str)
    parser.add_argument("output_file", type=str)

    # parse CLI args
    args = parser.parse_args()
    print(args)

    # run app
    main(args)
