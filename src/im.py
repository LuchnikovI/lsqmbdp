"""Influence matrix"""

from typing import List
import jax.numpy as jnp
from jax import Array
from jax.random import KeyArray, split, normal

InfluenceMatrix = List[Array]

InfluenceMatrixParameters = List[Array]


def random_params_weak_decay(
        subkey: KeyArray,
        local_choi_rank: int,
        sqrt_bond_dim: int,
) -> InfluenceMatrixParameters:
    """Generates isometric matrices that parametrize an influence matrix.
    This parametrization leads to weak decay.
    Args:
        subkey: jax random seed
        local_choi_rank: local choi rank
        sqrt_bond_dim: square root of bond dimension
    Returns: Influence matrix parameters"""

    def gen_random_isom(
            subkey: KeyArray,
            out_dim: int,
            inp_dim: int,
    ) -> Array:
        aux_dim = int(out_dim / inp_dim)
        ker = normal(subkey, (inp_dim, inp_dim, 2))
        ker = ker[..., 0] + 1j * ker[..., 1]
        ker, _ = jnp.linalg.qr(ker)
        ker = ker[jnp.newaxis]
        _, subkey = split(subkey)
        aux = normal(subkey, (aux_dim - 1, inp_dim, inp_dim, 2))
        aux = aux[..., 0] + 1j * aux[..., 1]
        aux = 0.01 * aux
        ker = jnp.concatenate([ker, aux], axis=0)
        ker = ker.reshape((out_dim, inp_dim))
        w, _, vh = jnp.linalg.svd(ker, full_matrices=False)
        ker = w @ vh
        return ker
    out_dims = 2 * [2 * sqrt_bond_dim * local_choi_rank]
    inp_dims = [2 * sqrt_bond_dim] + [2]
    subkeys = split(subkey, 2)
    params = [gen_random_isom(subkey, out_dim, inp_dim)\
              for subkey, out_dim, inp_dim in zip(subkeys, out_dims, inp_dims)]
    return params


def random_params(
        subkey: KeyArray,
        local_choi_rank: int,
        sqrt_bond_dim: int,
) -> InfluenceMatrixParameters:
    """Generates isometric matrices that parametrize an influence matrix.
    Args:
        subkey: jax random seed
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
    out_dims = 2 * [2 * sqrt_bond_dim * local_choi_rank]
    inp_dims = [2 * sqrt_bond_dim] + [2]
    subkeys = split(subkey, 2)
    params = [gen_random_isom(subkey, out_dim, inp_dim)\
              for subkey, out_dim, inp_dim in zip(subkeys, out_dims, inp_dims)]
    return params


def params2im(
        params: InfluenceMatrixParameters,
        time_steps: int,
        local_choi_rank: int,
) -> InfluenceMatrix:
    """Transforms parameters of an influence matrix into the influence matrix.
    Args:
        params: isometric matrices that parametrize an influance matrix
        time_steps: number of time steps
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
    influence_matrix = (time_steps - 1) * [translate_ker(params[0])] + [translate_ker(params[1])]
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

    params = random_params(key, local_choi_rank, sqrt_bond_dim)
    kers = params2im(params, time_steps, local_choi_rank)
    return kers


def id_im(time_steps: int):
    """Generates an Influance Matrix that does not affect dynamics
    of the system (identity influence matrix).
    Args:
        time_steps: number of time steps
    Returns: the identity influence matrix
    """

    kers = time_steps * [jnp.eye(4, dtype=jnp.complex64).reshape((1, 2, 2, 2, 2, 1))]
    return kers


def swap_and_phi_im(time_steps: int, phi: Array):
    """Generates a SWAP and transform influence matrix. Testing the only purpose of
    this influence matrix.
    Args:
        time_steps: number of time steps
    Returns: the SWAP and transform influence matrix
    """

    swap_phi = jnp.tensordot(jnp.eye(4), jnp.eye(4), axes=0).reshape((2, 2, 2, 2, 2, 2, 2, 2))
    swap_phi = swap_phi.transpose((0, 4, 1, 5, 2, 6, 3, 7))
    swap_phi = swap_phi.reshape((4, 2, 2, 2, 2, 4))
    swap_phi = jnp.tensordot(phi, swap_phi, axes=1)
    right = jnp.tensordot(swap_phi, jnp.eye(2).reshape((-1,)), axes=1)[..., jnp.newaxis]
    left = jnp.tensordot(jnp.eye(2).reshape((-1,)), swap_phi, axes=1)[jnp.newaxis]
    return [left] + (time_steps - 2) * [swap_phi] + [right]


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

    def trace_out(left: Array, ker: Array) -> Array:
        return jnp.tensordot(left, jnp.einsum("qiijjp->qp", ker), axes=1)
    left_states = [jnp.ones((1,))]
    for ker in influence_matrix[:-1]:
        left_state = trace_out(left_states[-1], ker)
        left_state /= jnp.linalg.norm(left_state)
        left_states.append(left_state)
    right_state = jnp.array([[1, 0, 0, 0]], dtype=jnp.complex64)
    rhos = [right_state.reshape((2, 2))]
    for ker, phi in zip(reversed(influence_matrix), reversed(phis)):
        left_state = left_states.pop()
        left_bond, _, _, _, _, right_bond = ker.shape
        ker = ker.reshape((left_bond, 4, 4, right_bond))
        right_state = jnp.tensordot(phi, right_state, axes=[[1], [1]])
        right_state = jnp.tensordot(ker, right_state, axes=2)
        rho = jnp.tensordot(left_state, right_state, axes=1).reshape((2, 2))
        norm = jnp.trace(rho)
        rho /= norm
        right_state /= norm
        rhos.append(rho)
    return rhos


def coupled_dynamics(
        influence_matrix1: InfluenceMatrix,
        influence_matrix2: InfluenceMatrix,
        int_gate: Array,
) -> List[Array]:
    """Computes dynamics of two coupled spins where each one is additionally
    coupled with its own environment.
    Args:
        influence_matrix1: influence matrix coupled with the first spin
        influence_matrix2: influence matrix coupled with the second spin
        int_gate: gate describing interaction between spins
    Returns:
        list of density matrices of these two spins evolving in time"""

    def trace_out(left: Array, ker: Array) -> Array:
            return jnp.tensordot(left, jnp.einsum("qiijjp->qp", ker), axes=1) / 2
    left_states1 = [jnp.ones((1,))]
    for ker in influence_matrix1[:-1]:
        left_state = trace_out(left_states1[-1], ker)
        left_states1.append(left_state)
    left_states2 = [jnp.ones((1,))]
    for ker in influence_matrix2[:-1]:
        left_state = trace_out(left_states2[-1], ker)
        left_states2.append(left_state)
    right_state = jnp.array([
        1, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    ]).reshape((1, 4, 4, 1))
    rhos = [right_state.reshape((2, 2, 2, 2)).transpose((0, 2, 1, 3)).reshape((4, 4))]
    first_matrix = True
    for ker1, ker2 in zip(reversed(influence_matrix1), reversed(influence_matrix2)):
        left_state1 = left_states1.pop()
        left_state2 = left_states2.pop()
        left_bond1, _, _, _, _, right_bond1 = ker1.shape
        ker1 = ker1.reshape((left_bond1, 4, 4, right_bond1))
        left_bond2, _, _, _, _, right_bond2 = ker2.shape
        ker2 = ker2.reshape((left_bond2, 4, 4, right_bond2))
        right_state = jnp.einsum("jkqp,iqpl->ijkl", int_gate, right_state)
        right_state = jnp.einsum("ijpq,qpkl->ijkl", ker1, right_state)
        right_state = jnp.einsum("lkqp,ijqp->ijkl", ker2, right_state)
        rho = jnp.einsum("q,p,qijp->ij", left_state1, left_state2, right_state)
        rho = rho.reshape((2, 2, 2, 2)).transpose((0, 2, 1, 3)).reshape((4, 4))
        # This is necessary to correct a normalization of im if it is incorrect
        if first_matrix:
            norm = jnp.trace(rho)
            first_matrix = False
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

    unitary = normal(subkey, (sq_dim, sq_dim, 2))
    unitary = unitary[..., 0] + 1j * unitary[..., 1]
    unitary, _ = jnp.linalg.qr(unitary)
    phi = jnp.tensordot(unitary, unitary.conj(), axes=0)
    phi = phi.transpose((0, 2, 1, 3))
    phi = phi.reshape((sq_dim * sq_dim, sq_dim * sq_dim))
    return phi
