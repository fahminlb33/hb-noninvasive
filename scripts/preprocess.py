import argparse

import xarray as xr

from rich import print as print_pretty
from magic.preprocessor import TransformComposeConfig, preprocess


# ----------------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------------


def main(args):
    # open dataset
    ds_hb = xr.open_dataset(args.dataset_file)

    # parse profile
    profile = TransformComposeConfig(**vars(args))
    print_pretty(profile)

    # extract data
    X, y = preprocess(ds_hb, profile)

    # create new dataset
    ds_rev = xr.Dataset(
        data_vars={
            "signal": (["id", "features"], X),
            "hb": (["id"], y),
        },
    )

    print("")
    print_pretty("Dataset created")
    print_pretty(ds_rev)

    # save to file
    ds_rev.to_netcdf(args.output_file)


if __name__ == "__main__":
    # create CLI parser
    parser = argparse.ArgumentParser()

    # add dataset and output paths
    parser.add_argument("dataset_file", type=str)
    parser.add_argument("output_file", type=str)

    # add parameters
    for field_name, hint in TransformComposeConfig.model_fields.items():
        parser.add_argument(f"--{field_name}", default=hint.default)

    # parse CLI args
    args = parser.parse_args()
    print(args)

    # run app
    main(args)
