"""Sampler"""

from functools import partial
from typing import List
from jax import Array, vmap
from jax.random import KeyArray, split, categorical
import jax.numpy as jnp
from im import InfluenceMatrix
from constants import projs
from sampler_utils import _build_left_states, _push_to_left

REGULARIZER = 1e-6

Sampler = List[Array]

def im2sampler(
        influence_matrix: InfluenceMatrix,
) -> Sampler:
    """Turns influence matrix into a sampler.
    Args:
        influence_matrix: Influence matrix
    Return: a sampler"""

    def translate_ker(ker: Array) -> Array:
        left_bond, _, _, _, _, right_bond = ker.shape
        ker = jnp.tensordot(ker, projs, axes=[[1, 2], [3, 2]])
        ker = jnp.tensordot(ker, projs, axes=[[1, 2], [3, 2]])
        ker = ker.transpose((0, 2, 4, 3, 5, 1))
        ker = ker.reshape((left_bond, 4, 16, right_bond)) / 2.
        return ker
    sampler = [translate_ker(ker) for ker in influence_matrix]
    return sampler


@partial(vmap, in_axes=(0, None, 0))
def _gen_samples(
        subkey: KeyArray,
        sampler: Sampler,
        indices: Array,
) -> Array:
    left_states = _build_left_states(sampler, indices)
    size = len(sampler)
    right_state = jnp.ones((1,))
    subkeys = split(subkey, size)
    idx = size - 1
    samples = jnp.zeros((size,))
    while len(left_states) != 0:
        left_state = left_states.pop()
        ker = sampler[idx]
        ker = ker[:, :, indices[idx]]
        ker = jnp.tensordot(left_state, ker, axes=1)
        ker = jnp.tensordot(ker, right_state, axes=1)
        sample = categorical(subkeys[idx], jnp.log(ker.real + REGULARIZER), shape=(1,))
        samples = samples.at[idx].set(sample[0])
        right_state = _push_to_left(right_state, sampler[idx], indices[idx], sample[0])
        norm = jnp.linalg.norm(right_state)
        right_state /= norm
        idx -= 1
    return samples


def gen_samples(
        subkey: KeyArray,
        sampler: Sampler,
        indices: Array,
) -> Array:
    """Produces samples from a sampler.
    Args:
        subkey: jax random seed
        sampler: Sampler
        indices: a matrix with indices each of that runs from [0 to 16)
            indicating the number of measurement basis
    Returns: set of sampler"""

    size = indices.shape[0]
    subkey = split(subkey, size)
    return _gen_samples(subkey, sampler, indices)
