import os
import json
import argparse

import tqdm

import pandas as pd
import xarray as xr

from scipy.stats import pearsonr
from matplotlib.figure import Figure

from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, PoissonRegressor, GammaRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)

from magic.preprocessor import preprocess, TransformComposeConfig

CONST_K_FOLD = 5

def create_model(name: str):
    # machine learning
    if name == "knn":
        return KNeighborsRegressor(n_jobs=2)
    elif name == "decision-tree":
        return DecisionTreeRegressor(random_state=42, max_depth=100)
    elif name == "random-forest":
        return RandomForestRegressor(random_state=42, max_depth=100, n_jobs=2)
    elif name == "gradient-boosting":
        return HistGradientBoostingRegressor(random_state=42, max_depth=100)
    elif name == "svr":
        return SVR()
    elif name == "mlp":
        return MLPRegressor(random_state=42)
    
    # statistical
    if name == "linear-regression":
        return LinearRegression(n_jobs=2)
    elif name == "ridge":
        return Ridge(random_state=42)
    elif name == "lasso":
        return Lasso(random_state=42)
    elif name == "elasticnet":
        return ElasticNet(random_state=42)
    elif name == "poisson":
        return PoissonRegressor()
    elif name == "gamma":
        return GammaRegressor()
    
    raise ValueError("Algorithm is not valid!")


def main(args):
    # load dataset
    ds_hb = xr.open_dataset(args.dataset_file)

    # preprocess
    profile = TransformComposeConfig(**vars(args))
    X, y = preprocess(ds_hb, profile)

    # setup cross-val
    cv = KFold(n_splits=CONST_K_FOLD, shuffle=True, random_state=21)
    # cv = RepeatedKFold(n_splits=CONST_K_FOLD, n_repeats=10, random_state=42)

    # open output file
    with open(args.output_file, "a+") as output_file:
        # perform cross-val
        for i, (train_index, test_index) in enumerate(
            tqdm.tqdm(cv.split(X, y), total=CONST_K_FOLD)
        ):
            # get data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # create model
            model = create_model(args.algorithm)

            # fit model
            model.fit(X_train, y_train)

            # predict
            y_pred = model.predict(X_test)

            # eval
            r, _ = pearsonr(y_test, y_pred)
            metrics = {
                "name": args.name,
                "algorithm": args.algorithm,
                "fold": i,
                "r": r,
                "r2": r2_score(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": root_mean_squared_error(y_test, y_pred),
            }

            # save metrics
            json.dump(
                metrics,
                output_file,
            )
            output_file.write("\n")

            # check if plot path is not null and exists
            if not args.plot_path or not os.path.exists(args.plot_path):
                return
            
            # create prediction dataframe
            df_pred = pd.DataFrame(
                {
                    "actual": y_test,
                    "predicted": y_pred,
                }
            ).sort_values("actual")

            # create figures
            fig = Figure(figsize=(8, 3))
            ax = fig.subplots(1, 2)

            # plot actual vs predicted
            ax[0].scatter(y_test, y_pred, alpha=0.5)
            ax[0].set_title("Actual vs Predicted")
            ax[0].set_xlabel("Hb predicted")
            ax[0].set_ylabel("Hb gold")

            # plot Hb trend
            ax[1].scatter(
                range(df_pred.shape[0]),
                df_pred["actual"],
                c="b",
                alpha=0.7,
                label="Actual",
            )
            ax[1].scatter(
                range(df_pred.shape[0]),
                df_pred["predicted"],
                c="r",
                alpha=0.2,
                label="Predicted",
            )
            ax[1].set_title("Actual vs Predicted Trend")
            ax[1].set_xlabel("Sample number")
            ax[1].set_ylabel("Hb value")
            ax[1].legend()

            # set layout and figure
            fig.tight_layout()
            fig.savefig(os.path.join(args.plot_path, f"{args.name}-{i+1}.png"))


if __name__ == "__main__":
    # set matplotlib backend
    import matplotlib

    matplotlib.use("Agg")

    # create CLI parser
    parser = argparse.ArgumentParser()

    # add dataset and output paths
    parser.add_argument("dataset_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--plot_path", type=str)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
    )

    # add parameters
    for field_name, hint in TransformComposeConfig.model_fields.items():
        parser.add_argument(f"--{field_name}", default=hint.default)

    # parse CLI args
    args = parser.parse_args()
    print(args)

    main(args)
