import re
import glob
import argparse

import tqdm

import numpy as np
import pandas as pd
import xarray as xr

from rich import print as print_pretty


def main(args):
    dataset = {"id": [], "name": [], "device": [], "hb": [], "signal": []}

    # read gold reading
    df_ref = pd.read_csv(args.reference_file)
    ref_map = {str(row.num): row.hb for row in df_ref.itertuples()}

    # find all relevant files
    for csv_file in glob.glob(args.input_path_glob):
        # get device number
        hb_num = re.search("HB[- ]?(\d{1,2})", csv_file, flags=re.IGNORECASE).group(1)

        # open CSV
        with open(csv_file, "r") as f:
            for line in tqdm.tqdm(
                f.readlines(), desc=f"Processing: HB-{hb_num}", position=0
            ):
                # split by |
                buf = line.split("|")

                # if the length is less than 10, this is empty recording
                if len(buf) < 10:
                    print_pretty(
                        f"[bold magenta]MISSING DATA![/bold magenta] No signal data was found. [bold magenta]SOURCE>[/bold magenta] HB-{hb_num}: ({buf[0]}) {buf[1].upper().strip()}"
                    )
                    continue

                if buf[-1] == "\n":
                    buf = buf[:-1]

                # extract signal
                signal = np.array([float(x) for x in buf[6:]])

                # assert signal length
                if signal.shape[0] != 8800:
                    print_pretty(
                        f"[bold magenta]MALFORMED DATA![/bold magenta] Unexpected signal length of [bright_yellow]{signal.shape[0]}[/bright_yellow] instead of [cyan]8800[/cyan]. [bold magenta]SOURCE>[/bold magenta] HB-{hb_num}: ({buf[0]}) {buf[1].upper().strip()}"
                    )
                    continue

                # scale data
                if args.scale_adc:
                    signal = signal / 65535.0  # 16 bit ADC

                # save to dataset
                dataset["id"].append(int(buf[0]))
                dataset["name"].append(buf[1].upper().strip())
                dataset["device"].append(int(hb_num))
                dataset["hb"].append(ref_map[buf[0]])
                dataset["signal"].append(signal)

    # create Dataset
    ds = xr.Dataset(
        data_vars={
            "signal": (["id", "time"], np.array(dataset["signal"])),
            "hb": (["id"], np.array(dataset["hb"])),
        },
        coords={
            "id": dataset["id"],
            "device": ("id", dataset["device"]),
            "name": ("id", dataset["name"]),
        },
    )

    # print summary
    print()
    print("Dataset created!")
    print_pretty(ds)

    # group by device
    counts = np.unique_counts(dataset["device"])
    counts_sr = pd.Series(counts.counts, index=counts.values)

    print()
    print("Patient count by device")
    print_pretty(counts_sr)

    # save to netCDF
    ds.to_netcdf(args.output_file)


if __name__ == "__main__":
    # create CLI parser
    parser = argparse.ArgumentParser()

    # add dataset and output paths
    parser.add_argument("input_path_glob", type=str)
    parser.add_argument("reference_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--scale_adc", action="store_true")

    # parse CLI args
    args = parser.parse_args()
    print(args)

    # run app
    main(args)
