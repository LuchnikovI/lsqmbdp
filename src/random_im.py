#!/usr/bin/env python3.10

import sys
import os
import h5py # type: ignore
from argparse import ArgumentParser
from im import random_im
from jax.random import split, PRNGKey

def main(argv):
    script_path = os.path.dirname(argv[0])
    sq_bond_dim = int(os.environ.get("SQ_BOND_DIM"))
    time_steps = int(os.environ.get("TIME_STEPS"))
    local_choi_rank = int(os.environ.get("LOCAL_CHOI_RANK"))
    seed = int(os.environ.get("SEED"))
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', default = "random_im")
    args = parser.parse_args()
    # -----------------------------------------------------------------------------
    key = PRNGKey(seed)
    _, subkey = split(key)
    infuence_matrix = random_im(subkey, time_steps, local_choi_rank, sq_bond_dim)
    hf = h5py.File(script_path + "/../shared_dir/" + args.name, 'w')
    group = hf.create_group("im")
    for i, ker in enumerate(infuence_matrix):
        group.create_dataset(str(i), data=ker)
    hf.close()

if __name__ == '__main__':
    main(sys.argv)