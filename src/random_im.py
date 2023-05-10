#!/usr/bin/env python3.10

# pylint: skip-file

import sys
import os
import h5py # type: ignore
from argparse import ArgumentParser
import jax.numpy as jnp
from im import random_im
from jax.random import split, PRNGKey


SQ_BOND_DIM = int(str(os.environ.get("SQ_BOND_DIM")))
TIME_STEPS = int(str(os.environ.get("TIME_STEPS")))
LOCAL_CHOI_RANK = int(str(os.environ.get("LOCAL_CHOI_RANK")))
SEED = int(str(os.environ.get("SEED")))
SCRIPT_PATH = os.path.dirname(sys.argv[0])


def main():
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', default = "random_im")
    args = parser.parse_args()
    # -----------------------------------------------------------------------------
    key = PRNGKey(SEED)
    _, subkey = split(key)
    influence_matrix = random_im(subkey, TIME_STEPS, LOCAL_CHOI_RANK, SQ_BOND_DIM)
    hf = h5py.File(SCRIPT_PATH + "/../shared_dir/" + args.name, 'w')
    group = hf.create_group("im")
    for i, ker in enumerate(influence_matrix):
        group.create_dataset(str(i), data=ker)
    hf.close()

if __name__ == '__main__':
    main()