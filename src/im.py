"""Influence matrix"""

from typing import List
import jax.numpy as jnp
from jax import Array
from jax.random import KeyArray, split, normal

InfluenceMatrix = List[Array]

InfluenceMatrixParameters = List[Array]


def random_unitary_params(
        subkey: KeyArray,
        time_steps: int,
        local_choi_rank: int,
        sqrt_bond_dim: int,
) -> InfluenceMatrixParameters:
    """Generates isometric matrices that parametrize an influence matrix.
    Those isometric matrices lead to Choi rank 1.
    Args:
        subkey: jax random seed
        time_steps: number of time steps
        local_choi_rank: local choi rank
        sqrt_bond_dim: square root of bond dimension
    Returns: Influence matrix parameters"""

    def gen_random_isom(
            subkey: KeyArray,
            out_dim: int,
            inp_dim: int,
            local_choi_rank: int,
    ) -> Array:
        ker = normal(subkey, (int(out_dim / local_choi_rank), inp_dim, 2))
        ker = ker[..., 0] + 1j * ker[..., 1]
        ker, _ = jnp.linalg.qr(ker)
        aux = jnp.zeros((local_choi_rank,))
        aux = aux.at[0].set(1.)
        ker = jnp.tensordot(aux, ker, axes=0)
        ker = ker.reshape((out_dim, inp_dim))
        return ker
    out_dims = time_steps * [2 * sqrt_bond_dim * local_choi_rank]
    inp_dims = (time_steps - 1) * [2 * sqrt_bond_dim] + [2]
    subkeys = split(subkey, time_steps)
    params = [gen_random_isom(subkey, out_dim, inp_dim, local_choi_rank)\
              for subkey, out_dim, inp_dim in zip(subkeys, out_dims, inp_dims)]
    return params


def random_params(
        subkey: KeyArray,
        time_steps: int,
        local_choi_rank: int,
        sqrt_bond_dim: int,
) -> InfluenceMatrixParameters:
    """Generates isometric matrices that parametrize an influence matrix.
    Args:
        subkey: jax random seed
        time_steps: number of time steps
        local_choi_rank: local choi rank
        sqrt_bond_dim: square root of bond dimension
    Returns: Influence matrix parameters"""

    def gen_random_isom(
            subkey: KeyArray,
            out_dim: int,
            inp_dim: int,
    ) -> Array:
        ker = normal(subkey, (out_dim, inp_dim, 2))
        ker = ker[..., 0] + 1j * ker[..., 1]
        ker, _ = jnp.linalg.qr(ker)
        return ker
    out_dims = time_steps * [2 * sqrt_bond_dim * local_choi_rank]
    inp_dims = (time_steps - 1) * [2 * sqrt_bond_dim] + [2]
    subkeys = split(subkey, time_steps)
    params = [gen_random_isom(subkey, out_dim, inp_dim)\
              for subkey, out_dim, inp_dim in zip(subkeys, out_dims, inp_dims)]
    return params


def params2im(
        params: InfluenceMatrixParameters,
        local_choi_rank: int,
) -> InfluenceMatrix:
    """Transforms parameters of an influence matrix into the influence matrix.
    Args:
        params: isometric matrices that parametrize an influance matrix
        local_choi_rank: local choi rank
    Returns: Influence matrix"""

    def translate_ker(
            ker: Array,
    ) -> Array:
        total_out_dim, total_inp_dim = ker.shape
        inp_dim = int(total_inp_dim / 2)
        out_dim = int(total_out_dim / (2 * local_choi_rank))
        ker = ker.reshape((local_choi_rank, 2, out_dim, 2, inp_dim))
        ker = jnp.tensordot(ker, ker.conj(), axes=[[0], [0]])
        ker = ker.transpose((1, 5, 0, 4, 2, 6, 3, 7))
        ker = ker.reshape((out_dim ** 2, 2, 2, 2, 2, inp_dim ** 2))
        return ker
    influence_matrix = [translate_ker(ker) for ker in params]
    total_out_dim, total_inp_dim = params[0].shape
    out_dim = int(total_out_dim / (2 * local_choi_rank))
    inp_dim = int(total_inp_dim / 2)
    last_ker = influence_matrix[0]
    last_ker = last_ker.reshape((out_dim, out_dim, 2, 2, 2, 2, inp_dim ** 2))
    last_ker = last_ker.trace(axis1=0, axis2=1)[jnp.newaxis]
    influence_matrix[0] = last_ker
    return influence_matrix


def random_im(
        key: KeyArray,
        time_steps: int,
        local_choi_rank: int,
        sqrt_bond_dim: int,
) -> InfluenceMatrix:
    """Generates a random Influance Matrix.
    Args:
        key: jax random seed
        time_steps: number of time steps
        local_choi_rank: dimension of a space that traced out at each time step
        sqrt_bomd_dim: square root of the bond dim
    Returns: random influence matrix"""

    params = random_params(key, time_steps, local_choi_rank, sqrt_bond_dim)
    kers = params2im(params, local_choi_rank)
    return kers


def random_unitary_im(
        key: KeyArray,
        time_steps: int,
        local_choi_rank: int,
        sqrt_bond_dim: int,
) -> InfluenceMatrix:
    """Generates a random free of dissipation Influance Matrix.
    Args:
        key: jax random seed
        time_steps: number of time steps
        local_choi_rank: dimension of a space that traced out at each time step
        sqrt_bomd_dim: square root of the bond dim
    Returns: random influence matrix"""

    params = random_unitary_params(key, time_steps, local_choi_rank, sqrt_bond_dim)
    kers = params2im(params, local_choi_rank)
    return kers


def im2phi(
        influence_matrix: InfluenceMatrix,
) -> Array:
    """Transforms an influence matrix to a quantum channel.
    Args:
        influence_matrix: influence matrix
    Returns: quantum channel"""

    phi = jnp.ones((1, 1, 1, 1, 1), dtype=jnp.complex64)
    for ker in influence_matrix:
        phi = jnp.tensordot(phi, ker, axes=1)
        phi = phi.transpose((0, 4, 1, 5, 2, 6, 3, 7, 8))
        new_dim = phi.shape[0] * phi.shape[1]
        phi = phi.reshape((new_dim, new_dim, new_dim, new_dim, -1))
    return phi[..., 0]

def dynamics(
        influence_matrix: InfluenceMatrix,
        phis: List[Array],
) -> List[Array]:
    """Computes dynamics of a system coupled with an influence matrix.
    Args:
        influence_matrix: influence matrix
        phis: quantum channels applied to the system each time step
    Returns: density matrices of the system"""

    def trace_out(ker: Array, left: Array) -> Array:
        return jnp.tensordot(left, jnp.einsum("qiijjp->qp", ker), axes=1)
    left_states = [jnp.ones((1,))]
    for ker in influence_matrix[:-1]:
        left_state = trace_out(ker, left_states[-1])
        left_state /= jnp.linalg.norm(left_state)
        left_states.append(left_state)
    right_state = jnp.array([[1, 0, 0, 0]], dtype=jnp.complex64)
    rhos = []
    for ker, phi in zip(reversed(influence_matrix), reversed(phis)):
        left_state = left_states.pop()
        lb, _, _, _, _, rb = ker.shape
        ker = ker.reshape((lb, 4, 4, rb))
        right_state = jnp.tensordot(phi, right_state, axes=[[1], [1]])
        right_state = jnp.tensordot(ker, right_state, axes=2)
        rho = jnp.tensordot(left_state, right_state, axes=1).reshape((2, 2))
        norm = jnp.trace(rho)
        rho /= norm
        right_state /= norm
        rhos.append(rho)
    return rhos


def random_unitary_channel(
        sq_dim: int,
        subkey: KeyArray,
) -> Array:
    """Generates a random unitary quantum channel.
    Args:
        sq_dim: Hilbert space dimension
        subkey: jax random seed
    Returns: quantum channel"""
    u = normal(subkey, (sq_dim, sq_dim, 2))
    u = u[..., 0] + 1j * u[..., 1]
    u, _ = jnp.linalg.qr(u)
    phi = jnp.tensordot(u, u.conj(), axes=0)
    phi = phi.transpose((0, 2, 1, 3))
    phi = phi.reshape((sq_dim * sq_dim, sq_dim * sq_dim))
    return phi
