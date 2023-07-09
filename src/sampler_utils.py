"""Utility functions for the sampler."""

from typing import List
from jax import Array
from jax.random import KeyArray, categorical, split
import jax.numpy as jnp

Sampler = List[Array]

def _push_to_right(
        left_state: Array,
        ker: Array,
) -> Array:
    ker = ker.sum(1)
    left_state = jnp.tensordot(left_state, ker, axes=1)
    return left_state


def _push_to_left(
        right_state: Array,
        ker: Array,
        sample: Array,
) -> Array:
    ker = ker[:, sample, :]
    right_state = jnp.tensordot(ker, right_state, axes=1)
    return right_state


def _build_left_states(
        sampler: Sampler,
) -> List[Array]:
    left_states = [jnp.ones((1,))]
    for ker in sampler:
        left_state = _push_to_right(left_states[-1], ker)
        norm = jnp.linalg.norm(left_state)
        left_state /= norm
        left_states.append(left_state)
    return left_states[:-1]


def _get_solid_probability(
        sampler: Sampler,
) -> Array:
    probability = jnp.ones((1,))
    for ker in sampler:
        probability = jnp.tensordot(probability, ker, axes=1)
    return probability[..., 0]



def _sample_from_solid_probability(
        subkey: KeyArray,
        probability: Array
) -> Array:
    size = len(probability.shape)
    subkeys = split(subkey, size)
    idx = size - 1
    samples = jnp.zeros((size,), dtype=jnp.int32)
    for _ in range(size):
        probability = probability.reshape((-1, 16))
        marginal_prob = probability.sum(0)
        marginal_prob /= marginal_prob.sum()
        sample = categorical(subkeys[idx], jnp.log(marginal_prob.real), shape=(1,))
        probability = probability[:, sample[0]]
        samples = samples.at[idx].set(sample[0])
        idx -= 1
    return samples
