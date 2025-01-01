import json
import argparse
import itertools
import subprocess

from tqdm import tqdm
from joblib import Parallel, delayed

DATASET_FILE = "data/20241227-final/merged-20241227.nc"
METRICS_FILE = "data/20241227-final/metrics-20241227-ml.jsonl"
PROFILE_FILE = "data/20241227-final/profiles-20241227-ml.jsonl"

ALGORITHMS = ["knn", "decision-tree", "random-forest", "gradient-boosting", "svr", "mlp"]
# ALGORITHMS = ["linear-regression", "ridge", "lasso", "elasticnet", "poisson", "gamma"]

def run_train(profile_number: int, scenario: list[str]):
    # create CLI arguments
    run_name = f"profile-{profile_number}"
    parameters = [
        # command
        "python",
        "scripts/train.py",
        # dataset file
        DATASET_FILE,
        # output metrics file
        METRICS_FILE,
        # run name
        "--name",
        run_name,
        # ML algorithm
        "--algorithm",
        scenario[5],
        # preprocessing
        "--device_number",
        str(scenario[0]),
        "--recording_section",
        str(scenario[1]),
        "--baseline_correction",
        str(scenario[2]),
        "--select_topk_pearson",
        str(scenario[4]),
    ]

    if scenario[3] != "time_domain_features":
        parameters.extend(
            ["--time_domain_features", "False", f"--{scenario[3]}", "True"]
        )

    # run process
    subprocess.call(
        parameters,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        cwd="/home/fahmi/research/hb-noninvasive",
    )

    return {
        "profile": profile_number,
        "device_number": str(scenario[0]),
        "recording_section": str(scenario[1]),
        "baseline_correction": str(scenario[2]),
        "feature_extraction": str(scenario[3]),
        "select_topk_pearson": str(scenario[4]),
        "algorithm": scenario[5],
    }


def main():
    # all possible parameters
    devices = [-1, 2, 7, 9, 11, 12, 13, 15]
    sections = [-1, 0, 1]
    baseline = [True, False]
    # savgol_filter = [True, False]
    feature_extraction = [
        "time_domain_features",
        "time_domain_stats_features",
        "frequency_domain_features",
        "frequency_domain_stats_features",
    ]
    select_topk_pearson = [-1, 20]

    # cross-product of parameters
    all_scenario = list(
        itertools.product(
            devices,
            sections,
            baseline,
            feature_extraction,
            select_topk_pearson,
            ALGORITHMS,
        )
    )

    # (-1, -1, True, 'time_domain_features', -1, 'linear-regression')
    # open profile file
    with open(PROFILE_FILE, "a+") as f:
        # create jobs
        jobs = [
            delayed(run_train)(profile_number, scenario)
            for profile_number, scenario in enumerate(all_scenario)
        ]

        # create executor
        executor = Parallel(
            n_jobs=5, backend="loky", return_as="generator_unordered", verbose=0
        )(jobs)

        # process all jobs
        pbar = tqdm(total=len(jobs))
        for result in executor:
            # update progress
            pbar.set_description(f"Profile: {result['profile']}")
            pbar.update(1)

            # save arguments as profile
            json.dump(result, f)
            f.write("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Got CTRL+C, stopping...")
