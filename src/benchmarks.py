#!/usr/bin/env python3.10

# pylint: skip-file

import os
import time
from functools import partial
from jax.random import PRNGKey, split, categorical
from jax import pmap, local_device_count, devices, local_devices, Array, value_and_grad
import jax.numpy as jnp
import termtables as tt # type: ignore
from im import random_params, params2im, random_im, InfluenceMatrixParameters
from sampler import im2sampler, gen_samples, log_prob

def main():
    sq_bond_dim = int(os.environ.get("SQ_BOND_DIM"))
    time_steps = int(os.environ.get("TIME_STEPS"))
    samples_number = int(os.environ.get("SAMPLES_NUMBER"))
    seed = int(os.environ.get("SEED"))
    # ----------------------------------------------------------------
    data = [[sq_bond_dim ** 2, time_steps, samples_number, seed]]
    header = ["Bond dimension", "Time steps", "Samples number", "Seed"]
    print("\nBenchmarking parameters:")
    tt.print(data, header)
    # ----------------------------------------------------------------
    try:
        data = [["#{}".format(i), device] for i, device in enumerate(local_devices(backend='gpu'))]
        print("Local GPUs:")
        tt.print(data)
        print("All GPUs:")
        data = [["#{}".format(i), device] for i, device in enumerate(devices(backend='gpu'))]
        tt.print(data)
    except:
        print("No GPUs detected")
    print("Local CPUs:")
    data = [["#{}".format(i), device] for i, device in enumerate(local_devices(backend='cpu'))]
    tt.print(data)
    print("All CPUs:")
    data = [["#{}".format(i), device] for i, device in enumerate(devices(backend='cpu'))]
    tt.print(data)
    # ----------------------------------------------------------------
    key = PRNGKey(seed)
    local_choi_rank = 2
    influence_matrix = random_im(key, time_steps, local_choi_rank, sq_bond_dim)
    sampler = im2sampler(influence_matrix)
    key, subkey = split(key)
    params = list(map(lambda x: jnp.tile(x[jnp.newaxis], (local_device_count(), 1, 1)), random_params(subkey, time_steps, 4 * (sq_bond_dim ** 2), sq_bond_dim)))

    @pmap
    @value_and_grad
    def loss_and_grad(
            params: InfluenceMatrixParameters,
            indices: Array,
            samples: Array,
    ):
        im = params2im(params, 4 * (sq_bond_dim ** 2))
        s = im2sampler(im)
        return -log_prob(s, indices, samples)

    # parallelized sampler
    p_gen_samples = pmap(gen_samples, in_axes=(0, None, 0))
    key, subkey = split(key)
    indices = categorical(
        subkey,
        jnp.ones((16,)),
        shape=(local_device_count(), samples_number, time_steps),
    )
    subkeys = split(key, local_device_count())
    print("Sampling benchmarks:")

    start = time.time()
    samples = p_gen_samples(subkeys, sampler, indices).block_until_ready()
    end = time.time()
    sampling_time_w_compilation = end - start

    start = time.time()
    loss_val, grads = loss_and_grad(params, indices, samples)
    end = time.time()
    grad_time_w_compilation = end - start

    start = time.time()
    samples = p_gen_samples(subkeys, sampler, indices).block_until_ready()
    end = time.time()
    sampling_time_wo_compilation = end - start

    start = time.time()
    loss_val, grads = loss_and_grad(params, indices, samples)
    end = time.time()
    grad_time_wo_compilation = end - start
    data = [
        ["Sampling time w/ compilation time", float(f'{sampling_time_w_compilation:.2f}'), "seconds"],
        ["Gradient time w/ compilation time", float(f'{grad_time_w_compilation:.2f}'), "seconds"],
        ["Sampling time w/o compilation time", float(f'{sampling_time_wo_compilation:.2f}'), "seconds"],
        ["Gradient time w/o compilation time", float(f'{grad_time_wo_compilation:.2f}'), "seconds"],
    ]
    tt.print(data)
    print("Shape of the resulting array of samples: {}".format(samples.shape))
    print("Shape of the gradient kernels:")
    for ker in grads:
        print(ker.shape)
    print("Value of the loss functions: {}".format(loss_val[0]))


if __name__ == '__main__':
    main()
