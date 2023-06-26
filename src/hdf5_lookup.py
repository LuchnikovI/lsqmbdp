#!/usr/bin/python3.10

# pylint: skip-file

import h5py # type: ignore
import argparse
import os
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p")
    args = parser.parse_args()
    h5_path = args.path
    h5_dir = os.path.dirname(h5_path)
    with h5py.File(args.path) as f:
        for key in f.keys():
            with h5py.File(h5_dir + "/experiments/ims/" + key, 'w') as subf:
                for name in f[key].keys():
                    subf.create_dataset(name=name, data=np.array(f[key][name]))

if __name__ == "__main__":
    main()