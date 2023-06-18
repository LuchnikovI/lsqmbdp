"""Sampler"""

from functools import partial
from typing import List
from jax import Array, vmap
from jax.random import KeyArray, split, categorical
import jax.numpy as jnp
from im import InfluenceMatrix, InfluenceMatrixParameters
from constants import projs
from sampler_utils import _build_left_states, _push_to_left

REGULARIZER = 0

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
    samples = jnp.zeros((size,), dtype=jnp.int32)
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


@partial(vmap, in_axes=(None, 0, 0, None))
def _log_prob(
        params: InfluenceMatrixParameters,
        indices: Array,
        samples: Array,
        local_choi_dim: int,
) -> Array:
    time_steps = indices.shape[0]
    assert samples.shape[0] == time_steps
    state = jnp.ones((1, 1))
    log_abs = jnp.zeros((1,))
    params = (time_steps - 1) * [params[0]] + [params[1]]
    for (ker, index, sample) in zip(params[::-1], indices[::-1], samples[::-1]):
        idx0, idx1 = jnp.unravel_index(index, (4, 4))
        smp0, smp1 = jnp.unravel_index(sample, (2, 2))
        projs0 = projs[smp0, idx0]
        projs1 = projs[smp1, idx1]
        _, total_inp_dim = ker.shape
        inp_dim = int(total_inp_dim / 2)
        ker = ker.reshape((local_choi_dim, 2, -1, 2, inp_dim))
        conj_ker = ker.conj()
        conj_ker = conj_ker.transpose((2, 3, 1, 0, 4))
        conj_ker = conj_ker.reshape((-1, 4 * local_choi_dim, inp_dim))
        ker = jnp.tensordot(projs0, ker, axes=[[1], [1]])
        ker = jnp.tensordot(projs1, ker, axes=[[1], [3]])
        ker = ker.transpose((3, 0, 1, 2, 4))
        ker = ker.reshape((-1, 4 * local_choi_dim, inp_dim))
        state = jnp.tensordot(conj_ker, state, axes=[[2], [1]])
        state = jnp.tensordot(ker, state, axes=[[1, 2], [1, 2]])
        norm = jnp.linalg.norm(state)
        state /= norm
        log_abs += jnp.log(norm)
    log_abs += jnp.log(jnp.trace(state))
    return log_abs[0].real


def log_prob(
        params: InfluenceMatrixParameters,
        indices: Array,
        samples: Array,
        local_choi_dim: int,
) -> Array:
    """Computes a logarithmic probability of measurements outcomes.
    Args:
        params: parameters of an influence matrix
        indices: measurement basis
        samples: measurement outcomes
        local_choi_dim: local Choi rank
    Returns: logarithm of probability"""

    return _log_prob(params, indices, samples, local_choi_dim).sum(0)
