"""Sampler"""

from functools import partial
from typing import List
from jax import Array, vmap
from jax.random import KeyArray, split, categorical
from jax.lax import dynamic_slice
import jax.numpy as jnp
from im import InfluenceMatrix, InfluenceMatrixParameters
from constants import povm
from sampler_utils import _build_left_states, _push_to_left

REGULARIZER = 0

Sampler = List[Array]

def im_log_norm(
        influence_matrix: InfluenceMatrix,
) -> Array:
    """Returns the normalization facto of an influence matrix.
    Args:
        influence_matrix: Influence matrix
    Returns: the normalization factor"""

    log_norm = jnp.zeros((1,))
    state = jnp.ones((1,))
    for ker in influence_matrix[::-1]:
        state = jnp.tensordot(jnp.einsum("iqqppj->ij", ker), state, axes=1)
        norm = jnp.linalg.norm(state)
        state /= norm
        log_norm += jnp.log(norm)
    return log_norm[0] + jnp.log(state[0]) - len(influence_matrix) * jnp.log(2)


def im2sampler(
        influence_matrix: InfluenceMatrix,
) -> Sampler:
    """Turns influence matrix into a sampler.
    Args:
        influence_matrix: Influence matrix
    Return: a sampler"""

    norm_per_ker = jnp.exp(im_log_norm(influence_matrix) / len(influence_matrix))
    def translate_ker(ker: Array) -> Array:
        left_bond, _, _, _, _, right_bond = ker.shape
        ker = jnp.tensordot(ker, povm, axes=[[1, 2], [2, 1]])
        ker = jnp.tensordot(ker, povm, axes=[[1, 2], [2, 1]])
        ker = ker.transpose((0, 2, 3, 1))
        ker = ker.reshape((left_bond, 16, right_bond)) / (2 * norm_per_ker)
        return ker
    sampler = [translate_ker(ker) for ker in influence_matrix]
    return sampler


@partial(vmap, in_axes=(0, None))
def _gen_samples(
        subkey: KeyArray,
        sampler: Sampler,
) -> Array:
    size = len(sampler)
    left_states = _build_left_states(sampler)
    right_state = jnp.ones((1,))
    subkeys = split(subkey, size)
    idx = size - 1
    samples = jnp.zeros((size,), dtype=jnp.int8)
    while len(left_states) != 0:
        left_state = left_states.pop()
        ker = sampler[idx]
        ker = jnp.tensordot(left_state, ker, axes=1)
        ker = jnp.tensordot(ker, right_state, axes=1)
        ker /= ker.sum()
        sample = categorical(subkeys[idx], jnp.log(ker.real), shape=(1,))
        samples = samples.at[idx].set(sample[0])
        right_state = _push_to_left(right_state, sampler[idx], sample[0])
        norm = jnp.linalg.norm(right_state)
        right_state /= norm
        idx -= 1
    return samples


def gen_samples(
        subkey: KeyArray,
        sampler: Sampler,
        size: int,
) -> Array:
    """Produces samples from a sampler.
    Args:
        subkey: jax random seed
        sampler: Sampler
        size: number of samples
    Returns: set of sampler"""

    subkey = split(subkey, size)
    return _gen_samples(subkey, sampler)


@partial(vmap, in_axes=(None, 0, None))
def _log_prob(
        params: InfluenceMatrixParameters,
        samples: Array,
        local_choi_dim: int,
) -> Array:
    state = jnp.ones((1,))
    log_abs = jnp.zeros((1,))
    _, total_inp_dim = params[0].shape
    inp_dim = int(total_inp_dim / 2)
    right_param = params[1].reshape((local_choi_dim, 2, inp_dim, 2, 1))
    mid_param = params[0].reshape((local_choi_dim, 2, inp_dim, 2, inp_dim))
    right_ker = jnp.einsum("qpirj,qsktl,asp,btr->abikjl", right_param, right_param.conj(), povm, povm)
    mid_ker = jnp.einsum("qpirj,qsktl,asp,btr->abikjl", mid_param, mid_param.conj(), povm, povm)
    right_ker = right_ker / 2
    mid_ker = mid_ker / 2
    for i, sample in enumerate(samples[::-1]):
        smp0, smp1 = jnp.unravel_index(sample, (4, 4))
        if i == 0:
            ker = right_ker[smp0, smp1]
            ker = ker.reshape((inp_dim ** 2, 1))
        else:
            ker = mid_ker[smp0, smp1]
            ker = ker.reshape((inp_dim ** 2, inp_dim ** 2))
        state = ker @ state
        norm = jnp.linalg.norm(state)
        state /= norm
        log_abs += jnp.log(norm)
    log_abs += jnp.log(jnp.trace(state.reshape(inp_dim, inp_dim)))
    return log_abs[0].real


@partial(vmap, in_axes=(None, 0))
def _log_prob_from_sampler(
        smpl: Sampler,
        samples: Array,
) -> Array:
    time_steps = samples.shape[0]
    assert samples.shape[0] == time_steps
    state = jnp.ones((1,))
    log_abs = jnp.zeros((1,))
    for (ker, sample) in zip(smpl[::-1], samples[::-1]):
        ker = dynamic_slice(ker,
                            (jnp.array(0, dtype=jnp.int8), sample, jnp.array(0, dtype=jnp.int8)),
                            (ker.shape[0], 1, ker.shape[-1]))[:, 0, :]
        state = jnp.tensordot(ker, state, axes=1)
        norm = jnp.linalg.norm(state)
        state /= norm
        log_abs += jnp.log(norm)
    return log_abs[0].real


def log_prob(
        params: InfluenceMatrixParameters,
        samples: Array,
        local_choi_dim: int,
) -> Array:
    """Computes a logarithmic probability of measurements outcomes.
    Args:
        params: parameters of an influence matrix
        samples: measurement outcomes
        local_choi_dim: local Choi rank
    Returns: logarithm of probability"""

    return _log_prob(params,  samples, local_choi_dim).sum(0)


def log_prob_from_sampler(
        smpl: Sampler,
        samples: Array,
) -> Array:
    """Computes a logarithmic probability of measurements outcomes from a sampler.
    Args:
        params: parameters of an influence matrix
        samples: measurement outcomes
        local_choi_dim: local Choi rank
    Returns: logarithm of probability"""

    return _log_prob_from_sampler(smpl, samples).sum(0)
