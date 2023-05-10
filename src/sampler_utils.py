"""Utility functions for the sampler."""

from typing import List
from jax import Array
from jax.random import KeyArray, categorical, split
import jax.numpy as jnp

REGULARIZER = 1e-6

Sampler = List[Array]

def _push_to_right(
        left_state: Array,
        ker: Array,
        index: Array,
) -> Array:
    ker = ker[:, :, index].sum(1)
    left_state = jnp.tensordot(left_state, ker, axes=1)
    return left_state


def _push_to_left(
        right_state: Array,
        ker: Array,
        index: Array,
        sample: Array,
) -> Array:
    ker = ker[:, sample, index, :]
    right_state = jnp.tensordot(ker, right_state, axes=1)
    return right_state


def _build_left_states(
        sampler: Sampler,
        indices: Array,
) -> List[Array]:
    left_states = [jnp.ones((1,))]
    for ker, index in zip(sampler, indices):
        left_state = _push_to_right(left_states[-1], ker, index)
        norm = jnp.linalg.norm(left_state)
        left_state /= norm
        left_states.append(left_state)
    return left_states[:-1]


def _get_solid_probability(
        sampler: Sampler,
        index: Array,
) -> Array:
    probability = jnp.ones((1,))
    for ker, i in zip(sampler, index):
        ker = ker[:, :, i]
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
        probability = probability.reshape((-1, 4))
        marginal_prob = probability.sum(0)
        sample = categorical(subkeys[idx], jnp.log(marginal_prob.real + REGULARIZER), shape=(1,))
        probability = probability[:, sample[0]]
        samples = samples.at[idx].set(sample[0])
        idx -= 1
    return samples
