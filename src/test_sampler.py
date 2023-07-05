"""Sampler tests."""

import pytest
import jax.numpy as jnp
from jax.random import PRNGKey, KeyArray, split, categorical
from sampler import (
    im2sampler,
    _gen_samples,
    log_prob,
    log_prob_from_sampler,
)
from im import (
    random_im,
    params2im,
    random_params,
)
from mpa import mpa_sum, mpa2tensor
from sampler_utils import (
    _get_solid_probability,
    _sample_from_solid_probability,
)

KEY = PRNGKey(47)

ACC = 1e-4

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


@pytest.mark.parametrize("subkey", split(KEY, 2))
@pytest.mark.parametrize("time_steps", [2, 5])
@pytest.mark.parametrize("local_choi_rank", [2, 4])
@pytest.mark.parametrize("sqrt_bond_dim", [1, 5])
@pytest.mark.parametrize("samples_number", [10])
def test_log_probabilities(
        subkey: KeyArray,
        time_steps: int,
        local_choi_rank: int,
        sqrt_bond_dim: int,
        samples_number: int,
):
    """Test logarithmic probabilities."""

    indices = categorical(subkey, jnp.ones((16,)), shape=(samples_number, time_steps))
    _, subkey = split(subkey)
    params = random_params(subkey, local_choi_rank, sqrt_bond_dim)
    influence_matrix = params2im(params, time_steps, local_choi_rank)
    for i in range(len(influence_matrix)):
        influence_matrix[i] = influence_matrix[i] / (1.5 + 2j)
    sampler = im2sampler(influence_matrix)
    subkey = split(subkey, samples_number)
    samples = _gen_samples(subkey, sampler, indices)
    log_prob_1 = log_prob(params, indices, samples, local_choi_rank)
    log_prob_2 = log_prob_from_sampler(sampler, indices, samples)
    assert jnp.abs(log_prob_1 - log_prob_2) < ACC
