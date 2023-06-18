"""Sampler tests."""

import pytest
import jax.numpy as jnp
from jax.random import PRNGKey, KeyArray, split, categorical
from sampler import im2sampler, _gen_samples
from im import random_im
from mpa import mpa_sum, mpa2tensor
from sampler_utils import (
    _get_solid_probability,
    _sample_from_solid_probability,
)

KEY = PRNGKey(47)

ACC = 1e-5

@pytest.mark.parametrize("subkey", split(KEY, 2))
@pytest.mark.parametrize("time_steps", [2, 5])
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


@pytest.mark.parametrize("subkey", split(KEY, 2))
@pytest.mark.parametrize("time_steps", [2, 5])
@pytest.mark.parametrize("local_choi_rank", [2, 4])
@pytest.mark.parametrize("sqrt_bond_dim", [1, 5])
def test_sampler(
        subkey: KeyArray,
        time_steps: int,
        local_choi_rank: int,
        sqrt_bond_dim: int,
):
    """Tests sampler comparing the MPS based version with the explicit one."""
    _, subkey = split(subkey)
    sampler = im2sampler(random_im(subkey, time_steps, local_choi_rank, sqrt_bond_dim))
    for _ in range(10):
        _, subkey = split(subkey)
        index = categorical(subkey, jnp.ones((16,)), shape=(time_steps,))
        _, subkey = split(subkey)
        probability = _get_solid_probability(sampler, index)
        solid_sample = _sample_from_solid_probability(subkey, probability)
        sample = _gen_samples(subkey[jnp.newaxis], sampler, index[jnp.newaxis])
        assert (sample[0] == solid_sample).all()
