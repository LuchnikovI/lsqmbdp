"""Matrix product array (MPA)."""
from typing import List, Tuple
import jax.numpy as jnp
from jax import Array
from jax.random import KeyArray, split
from utils import _random_normal_complex

MPA = List[Array]

def mpa_gen_random(
        key: KeyArray,
        shapes: List[List[int]],
        bond_dim: int,
) -> MPA:
    """Generates a random MPA.
    Args:
        key: jax random seed
        shapes: list of shapes of physical indices per side
        bond_dim: maximal bond dimension
    Returns: MPA"""

    size = len(shapes)
    subkeys = split(key, size)
    left_bonds = [1] + (size - 1) * [bond_dim]
    right_bonds = (size - 1) * [bond_dim] + [1]
    def gen_kernel(args):
        return _random_normal_complex(args[0], [args[1]] + args[2] + [args[3]])
    mpa = [gen_kernel(args)for args in zip(subkeys, left_bonds, shapes, right_bonds)]
    return mpa


def mpa2tensor(mpa: MPA) -> Array:
    """Turns MPA into a solid tensor.
    Args:
        mpa: MPA
    Returns: tensor representation of mpa"""
    left_state = jnp.ones((1,))
    for ker in mpa:
        left_state = jnp.tensordot(left_state, ker, axes=1)
    return left_state[..., 0]


def mpa_log_sum(mpa: MPA) -> Tuple[Array, Array]:
    """Computes the logarithm of sum of all elements of an MPA.
    Args:
        mpa: MPA
    Returns: log|MPA.sum|, exp(i * arg(MPA.sum))"""

    def _push_sum_right(
        left_state: Array,
        ker: Array,
    ) -> Array:
        left_bond = ker.shape[0]
        right_bond = ker.shape[-1]
        ker = ker.reshape((left_bond, -1, right_bond))
        ker = ker.sum(1)
        left_state = jnp.tensordot(left_state, ker, axes=1)
        return left_state
    left_state = jnp.ones((1,))
    log_norm = jnp.array(0.)
    for ker in mpa:
        left_state = _push_sum_right(left_state, ker)
        norm = jnp.linalg.norm(left_state)
        left_state /= norm
        log_norm += jnp.log(norm)
    return log_norm, left_state[0]


def mpa_sum(mpa: MPA) -> Array:
    """Computes the sum of all elements of an MPA.
    Args:
        mpa: MPA
    Returns: sum of all elements"""

    log_abs, exp_phase = mpa_log_sum(mpa)
    return jnp.exp(log_abs) * exp_phase


def mpa_log_dot(lhs_mpa: MPA, rhs_mpa: MPA) -> Tuple[Array, Array]:
    """Computes the logarithmic value of the dot product between two MPAs.
    Args:
        lhs_mpa: left side mpa
        rhs_mpa: right side mpa
    Returns: log|<lhs_mpa, rhs_mpa>|, exp(i * arg(<lhs_mpa, rhs_mpa>))"""

    def _push_dot_right(
        left_state: Array,
        ker_up: Array,
        ker_down: Array,
    ) -> Array:
        left_bond = ker_up.shape[0]
        right_bond = ker_up.shape[-1]
        ker_up = ker_up.reshape((left_bond, -1, right_bond))
        left_bond = ker_down.shape[0]
        right_bond = ker_down.shape[-1]
        ker_down = ker_down.reshape((left_bond, -1, right_bond))
        left_state = jnp.tensordot(left_state, ker_down.conj(), axes=[[0], [0]])
        left_state = jnp.tensordot(left_state, ker_up, axes=[[0, 1], [0, 1]])
        return left_state
    left_state = jnp.ones((1, 1))
    log_norm = jnp.array(0.)
    for ker_up, ker_down in zip(lhs_mpa, rhs_mpa):
        left_state = _push_dot_right(left_state, ker_up, ker_down)
        norm = jnp.linalg.norm(left_state)
        left_state /= norm
        log_norm += jnp.log(norm)
    return log_norm, left_state[0, 0]


def mpa_dot(lhs_mpa: MPA, rhs_mpa: MPA) -> Array:
    """Computes the value of the dot product between two MPAs.
    Args:
        lhs_mpa: left side mpa
        rhs_mpa: right side mpa
    Returns: dot product value"""

    log_abs, exp_phase = mpa_log_dot(lhs_mpa, rhs_mpa)
    return jnp.exp(log_abs) * exp_phase


def set_to_forward_canonical(mpa: MPA) -> Array:
    """Sets an MPA into the forward canonical form inplace.
    Args:
        mpa: MPA
    Returns: logarithm of norm of an MPA"""

    def _push_r_backward(
        ker: Array,
        rker: Array,
    ) -> Tuple[Array, Array]:
        left_bond = rker.shape[0]
        right_bond = ker.shape[-1]
        mid_shape = ker.shape[1:-1]
        ker = jnp.tensordot(rker, ker, axes=1)
        ker = ker.reshape((-1, right_bond))
        ker, rker = jnp.linalg.qr(ker)
        ker = ker.reshape((left_bond,) + mid_shape + (-1,))
        return ker, rker
    rker = jnp.eye(1)
    log_norm = jnp.array(0.)
    for i, ker in enumerate(mpa):
        ker, rker = _push_r_backward(ker, rker)
        norm = jnp.linalg.norm(rker)
        rker /= norm
        log_norm += jnp.log(norm)
        mpa[i] = ker
    mpa[-1] = jnp.tensordot(mpa[-1], rker, axes=1)
    return log_norm
