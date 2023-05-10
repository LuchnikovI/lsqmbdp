#!/usr/bin/env python3.10

import sys
import os
from functools import partial
import h5py # type: ignore
import numpy as np
import jax.numpy as jnp
from argparse import ArgumentParser
from jax.random import split, PRNGKey
from jax import pmap, value_and_grad, local_device_count, devices, Array
from jax.lax import pmean
from qgoptax.manifolds import StiefelManifold # type: ignore
from qgoptax.optimizers import RAdam # type: ignore
from sampler import log_prob, im2sampler
from im import InfluenceMatrixParameters, InfluenceMatrix, params2im, random_unitary_params
from mpa import set_to_forward_canonical, mpa_log_dot


SCRIPT_PATH = os.path.dirname(sys.argv[0])
LEARNING_RATE_IN = float(str(os.environ.get("LEARNING_RATE_IN")))
LEARNING_RATE_FINAL = float(str(os.environ.get("LEARNING_RATE_FINAL")))
SAMPLES_NUMBER = int(str(os.environ.get("SAMPLES_NUMBER")))
TOTAL_SAMPLES_NUMBER = int(str(os.environ.get("TOTAL_SAMPLES_NUMBER")))
SQ_BOND_DIM_TRAINING = int(str(os.environ.get("SQ_BOND_DIM_TRAINING")))
EPOCHS_NUMBER = int(str(os.environ.get("EPOCHS_NUMBER")))
SEED = int(str(os.environ.get("SEED")))
LOCAL_DEVICES_NUM = local_device_count()
EPOCH_SIZE = int(TOTAL_SAMPLES_NUMBER / (SAMPLES_NUMBER * LOCAL_DEVICES_NUM))
MAIN_CPU = devices("cpu")[0]
DECAY_COEFF = (LEARNING_RATE_FINAL / LEARNING_RATE_IN) ** (1 / EPOCH_SIZE)
LOCAL_CHOI_RANK_TRAINING = int(str(os.environ.get("LOCAL_CHOI_RANK_TRAINING")))


@pmap
@value_and_grad
def _loss_and_grad(
        params: InfluenceMatrixParameters,
        data: Array,
) -> Array:
    im = params2im(params, LOCAL_CHOI_RANK_TRAINING)
    s = im2sampler(im)
    return -log_prob(s, data[:, 0], data[:, 1])


@partial(pmap, axis_name='i')
def _av_grad(
        grads: InfluenceMatrixParameters,
) -> InfluenceMatrixParameters:
    return pmean(grads, axis_name='i')


def _hdf2im(
        path: str,
) -> InfluenceMatrix:
    with h5py.File(path, "r") as f:
        def idx2ker(idx: str):
            ker = jnp.array(f["im"][str(idx)])
            return ker
        kers_num = len(f["im"].values())
        return [idx2ker(idx) for idx in range(kers_num)]


def _hdf2data(
        path: str,
) -> Array:
    with h5py.File(path, "r") as f:
        data = np.array(f["data"])
    return data


par_random_unitary_params = pmap(random_unitary_params, static_broadcasted_argnums=(1, 2, 3))


def main():
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', default = f"random_im")
    args = parser.parse_args()
    # ---------------------------------------------------------------------------------
    unknown_influence_matrix = _hdf2im(SCRIPT_PATH + "/../shared_dir/" + args.name)
    set_to_forward_canonical(unknown_influence_matrix)
    time_steps = len(unknown_influence_matrix)
    data = _hdf2data(SCRIPT_PATH + "/../shared_dir/" + args.name + "_data")
    data = data.reshape((EPOCH_SIZE, LOCAL_DEVICES_NUM, SAMPLES_NUMBER, 2, time_steps))
    # ---------------------------------------------------------------------------------
    key = PRNGKey(SEED)
    key, _ = split(key)
    key, _ = split(key)
    key, _ = split(key)
    _, subkey = split(key)
    subkeys = jnp.tile(subkey, (LOCAL_DEVICES_NUM, 1))
    params = par_random_unitary_params(subkeys, time_steps, LOCAL_CHOI_RANK_TRAINING, SQ_BOND_DIM_TRAINING)
    man = StiefelManifold()
    lr = LEARNING_RATE_IN
    opt = RAdam(man, lr)
    opt_state = opt.init(params)
    hf = h5py.File(SCRIPT_PATH + "/../shared_dir/" + args.name, 'a')
    best_loss_val = jnp.finfo(jnp.float32).max
    for i in range(1, EPOCHS_NUMBER + 1):
        opt = RAdam(man, lr)
        av_loss_val = 0.
        for data_slice in data:
            loss_val, grads = _loss_and_grad(params, data_slice)
            grads = _av_grad(grads)
            params, opt_state = opt.update(grads, opt_state, params)
            av_loss_val += loss_val
        np.random.shuffle(data)
        lr *= DECAY_COEFF
        found_influence_matrix = params2im([ker[0] for ker in params], LOCAL_CHOI_RANK_TRAINING)
        if av_loss_val < best_loss_val:
            best_loss_val = av_loss_val
            del hf["im"]
            group = hf.create_group("im")
            for j, ker in enumerate(found_influence_matrix):
                group.create_dataset(str(j), data=ker)
        set_to_forward_canonical(found_influence_matrix)
        log_fidelity_half, _ = mpa_log_dot(unknown_influence_matrix, found_influence_matrix)
        fidelity = jnp.exp(2 * log_fidelity_half)
        av_loss_val = av_loss_val / EPOCH_SIZE
        print("Epoch num: {:<5} Loss value: {:<20} Fidelity: {}".format(i, float(av_loss_val), float(fidelity)))
    hf.close()

if __name__ == '__main__':
    main()
