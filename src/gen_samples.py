#!/usr/bin/env python3.10

# pylint: skip-file

import sys
import os
import h5py # type: ignore
import jax.numpy as jnp
from argparse import ArgumentParser
from jax.random import split, PRNGKey, categorical
from jax import pmap, local_device_count, devices, device_put
from sampler import gen_samples, im2sampler
from im import InfluenceMatrix

SAMPLES_NUMBER = int(str(os.environ.get("SAMPLES_NUMBER")))
TOTAL_SAMPLES_NUMBER = int(str(os.environ.get("TOTAL_SAMPLES_NUMBER")))
SEED = int(str(os.environ.get("SEED")))
SCRIPT_PATH = os.path.dirname(sys.argv[0])
MAIN_CPU = devices("cpu")[0]
LOCAL_DEVICES_NUM = local_device_count()
SAMPLING_EPOCHS_NUM = int(TOTAL_SAMPLES_NUMBER / (SAMPLES_NUMBER * LOCAL_DEVICES_NUM))

def _hdf2im(
        path: str,
) -> InfluenceMatrix:
    with h5py.File(path) as f:
        def idx2ker(idx: int):
            ker = jnp.array(f["im"][str(idx)])
            size = len(ker.shape)
            ker = ker[jnp.newaxis]
            ker = jnp.tile(ker, (LOCAL_DEVICES_NUM,) + size * (1,))
            return ker
        kers_num = len(f["im"].values())
        influence_matrix = [idx2ker(idx) for idx in range(kers_num)]
    return influence_matrix


par_im2sampler = pmap(im2sampler)


par_gen_samples = pmap(gen_samples, in_axes=(0, 0, 0))


def main():
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', default = f"random_im")
    args = parser.parse_args()
    # --------------------------------------------------------------------------------
    influence_matrix = _hdf2im(SCRIPT_PATH + "/../shared_dir/" + args.name)
    par_im2sampler = pmap(im2sampler)
    sampler = par_im2sampler(influence_matrix)
    time_steps = len(sampler)
    # ----------------------------------------------------------------------
    key = PRNGKey(SEED)
    key, _ = split(key)
    key, _ = split(key)
    keys = split(key, SAMPLING_EPOCHS_NUM)
    all_indices = device_put(jnp.zeros((0, time_steps), dtype=jnp.int32), device=MAIN_CPU)
    all_samples = device_put(jnp.zeros((0, time_steps), dtype=jnp.int32), device=MAIN_CPU)
    for key in keys:
        key, subkey = split(key)
        indices = categorical(subkey, jnp.ones((16,)), shape=(LOCAL_DEVICES_NUM, SAMPLES_NUMBER, time_steps))
        subkeys = split(key, LOCAL_DEVICES_NUM)
        samples = par_gen_samples(subkeys, sampler, indices)
        all_indices = jnp.concatenate([all_indices, device_put(indices, MAIN_CPU).reshape((-1, time_steps))])
        all_samples = jnp.concatenate([all_samples, device_put(samples, MAIN_CPU).reshape((-1, time_steps))])
    data = jnp.concatenate([all_indices[:, jnp.newaxis], all_samples[:, jnp.newaxis]], axis=1)
    with h5py.File(SCRIPT_PATH + "/../shared_dir/" + args.name + "_data", "w") as f:
        f.create_dataset("data", data=data)

if __name__ == '__main__':
    main()
