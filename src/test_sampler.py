"""Sampler tests."""

import pytest
import jax.numpy as jnp
from jax.random import PRNGKey, KeyArray, split, categorical
from sampler import im2sampler
from im import random_im
from mpa import mpa_sum, mpa2tensor

KEY = PRNGKey(42)

ACC = 1e-5

@pytest.mark.parametrize("subkey", split(KEY, 2))
@pytest.mark.parametrize("time_steps", [1, 5])
@pytest.mark.parametrize("local_choi_rank", [2, 4])
@pytest.mark.parametrize("sqrt_bond_dim", [1, 5])
def test_im2sampler(
        subkey: KeyArray,
        time_steps: int,
        local_choi_rank: int,
        sqrt_bond_dim: int,
):
    """Property test of a sampler."""

    _, subkey = split(subkey)
    sampler = im2sampler(random_im(subkey, time_steps, local_choi_rank, sqrt_bond_dim))
    _, subkey = split(subkey)
    index = categorical(subkey, jnp.ones((16,)), shape=(time_steps,))
    for j, i in enumerate(index):
        ker = sampler[j]
        ker = ker[:, :, i]
        sampler[j] = ker
    sum_val = mpa_sum(sampler)
    assert (jnp.abs(sum_val - 1.) < ACC).all()
    arr = mpa2tensor(sampler).reshape((-1,))
    assert (arr >= -ACC).all()
