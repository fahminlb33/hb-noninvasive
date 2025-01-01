from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from pydantic import BaseModel
from scipy.fft import fft
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
from scipy.signal.windows import blackman, hamming


# ----------------------------------------------------------------------------
# Config Models
# ----------------------------------------------------------------------------


class TransformComposeConfig(BaseModel):
    # pre-processing
    device_number: int = -1  # -1 = all device, otherwise select
    recording_section: int = -1  # -1 = all, 0 = slow, 1 = fast
    baseline_correction: bool = False
    baseline_correction_n: int = 1100
    savgol_filter: bool = False
    savgol_filter_window_length: int = 11
    savgol_filter_polynomial_order: int = 2
    savgol_filter_derivative_order: int = 1

    # feature extraction
    time_domain_features: bool = True
    time_domain_stats_features: bool = False
    time_domain_stats: list[Literal["mean", "std", "var", "rms"]] = [
        "mean",
        "std",
        "var",
        "rms",
    ]
    frequency_domain_features: bool = False
    frequency_domain_stats_features: bool = False
    frequency_domain_window: Literal["no-window", "hamming", "blackman"] = "no-window"

    # post processing
    select_topk_pearson: int = -1  # -1 = all features, otherwise select


# ----------------------------------------------------------------------------
# Compose Preprocessor
# ----------------------------------------------------------------------------


def derive_stats(values: np.ndarray, stats: list[str]):
    current_feat = []
    if "mean" in stats:
        current_feat.append(np.mean(values, axis=1))
    if "std" in stats:
        current_feat.append(np.std(values, axis=1))
    if "var" in stats:
        current_feat.append(np.var(values, axis=1))
    if "rms" in stats:
        current_feat.append(np.sqrt(np.mean(values**2, axis=1)))

    return np.vstack(current_feat).T


def derive_fft(values: np.ndarray, window: str):
    signal = values.copy()

    if window == "hamming":
        signal = signal * hamming(signal.shape[1])
    elif window == "blackman":
        signal = signal * blackman(signal.shape[1])

    return np.abs(fft(signal))[:, : signal.shape[1] // 2]


def preprocess(ds: xr.Dataset, config: TransformComposeConfig):
    # ----- PRE-PROCESSING

    # select device
    if config.device_number != -1:
        ds = ds.sel(id=ds["device"] == config.device_number)

    # get X and y
    X = ds["signal"].to_numpy()
    y = ds["hb"].to_numpy()

    # select recording phase
    if config.recording_section == 0:
        X = X[:4400]
        y = y[:4400]
    elif config.recording_section == 1:
        X = X[4400:]
        y = y[4400:]

    # perform baseline correction
    if config.baseline_correction:
        X = X[:, config.baseline_correction_n :] - np.mean(
            X[:, : config.baseline_correction_n]
        )

    # perform savitzky-golay filter
    if config.savgol_filter:
        X = savgol_filter(
            X,
            config.savgol_filter_window_length,
            config.savgol_filter_polynomial_order,
            config.savgol_filter_derivative_order,
        )

    # ----- FEATURE EXTRACTION

    # create empty dataset
    X_features = np.zeros_like(X)

    # copy data
    if config.time_domain_features:
        X_features = X
    elif config.time_domain_stats_features:
        X_features = derive_stats(X, config.time_domain_stats)
    elif config.frequency_domain_features:
        X_features = derive_fft(X, config.frequency_domain_window)

    # ----- POST-PROCESSING

    # select absolute top-k pearson correlation
    if config.select_topk_pearson != -1:
        # calculate Pearson correlation
        r, _ = pearsonr(X_features, y.reshape(-1, 1))

        # create Pandas series from absolute r-values
        sr_corr = pd.Series(np.abs(r), index=range(X.shape[1])).sort_values(
            ascending=False
        )

        # select top-k
        topk_indices = sr_corr.nlargest(config.select_topk_pearson).index.to_numpy()

        return X_features[:, topk_indices], y

    return X_features, y
