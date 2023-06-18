"""This is tests for influence matrix."""

from jax.random import KeyArray, PRNGKey, split
import jax.numpy as jnp
import pytest
from im import (
    random_im,
    im2phi,
    dynamics,
    coupled_dynamics,
    random_unitary_channel,
    swap_and_phi_im,
)

KEY = PRNGKey(42)

ACC = 1e-4

@pytest.mark.parametrize("subkey", split(KEY, 2))
@pytest.mark.parametrize("sqrt_dim", [1, 2, 4])
def test_random_unitary_channel(
        subkey: KeyArray,
        sqrt_dim: int,
):
    """Tests correctness of a generated unitary quantum channel"""
    phi = random_unitary_channel(sqrt_dim, subkey)
    choi = phi.reshape((sqrt_dim, sqrt_dim, sqrt_dim, sqrt_dim))
    choi = choi.transpose((0, 2, 1, 3))
    choi = choi.reshape((sqrt_dim ** 2, sqrt_dim ** 2))
    assert ((choi - choi.T.conj()) < ACC).all()
    lmbd = jnp.linalg.eigvalsh(choi)
    assert (lmbd > -ACC).all()
    assert (lmbd > ACC).sum() == 1


@pytest.mark.parametrize("subkey", split(KEY, 2))
@pytest.mark.parametrize("time_steps", [2, 5])
@pytest.mark.parametrize("local_choi_rank", [1, 3])
@pytest.mark.parametrize("sqrt_bond_dim", [1, 5])
def test_random_im(
        subkey: KeyArray,
        time_steps: int,
        local_choi_rank: int,
        sqrt_bond_dim: int,
):
    """Tests random_im function by building an explicit channel and testing its properties"""
    influance_matrix = random_im(subkey, time_steps, local_choi_rank, sqrt_bond_dim)
    right_bonds_sq = (len(influance_matrix) - 2) * [sqrt_bond_dim] + [1]
    left_bonds_sq = (len(influance_matrix) - 1) * [sqrt_bond_dim]
    for ker, left_bond_sq, right_bond_sq in\
    zip(influance_matrix[1:], left_bonds_sq, right_bonds_sq):
        ker = ker.reshape((left_bond_sq, left_bond_sq, 2, 2, 2, 2, right_bond_sq, right_bond_sq))
        ker = ker.transpose((0, 2, 4, 6, 1, 3, 5, 7))
        ker = ker.reshape((left_bond_sq * right_bond_sq * 4, left_bond_sq * right_bond_sq * 4))
        eigvalsh = jnp.linalg.eigvalsh(ker)
        print(eigvalsh)
        assert (jnp.abs(ker - ker.conj().T) < ACC).all()
        assert (eigvalsh > -ACC).all()
        assert (eigvalsh > ACC).sum() == local_choi_rank
    phi = im2phi(influance_matrix)
    dim = 2 ** time_steps
    assert phi.shape == (dim, dim, dim, dim)
    tr_phi = phi.trace(axis1=0, axis2=1)
    assert (jnp.abs(tr_phi -jnp.eye(dim)) < ACC).all()
    phi = phi.transpose((0, 2, 1, 3))
    phi = phi.reshape((dim * dim, dim * dim))
    assert (jnp.abs(phi - phi.conj().T) < ACC).all()
    assert (jnp.linalg.eigvalsh(phi) > -ACC).all()


@pytest.mark.parametrize("subkey", split(KEY, 2))
@pytest.mark.parametrize("time_steps", [2, 5])
@pytest.mark.parametrize("local_choi_rank", [1, 3])
@pytest.mark.parametrize("sqrt_bond_dim", [1, 5])
def test_dynamics(
        subkey: KeyArray,
        time_steps: int,
        local_choi_rank: int,
        sqrt_bond_dim: int,
):
    """Tests correctness of density matrices calculated within dynamics simulation"""
    subkeys = split(subkey, time_steps)
    phis = [random_unitary_channel(2, subkey) for subkey in subkeys]
    influance_matrix = random_im(subkey, time_steps, local_choi_rank, sqrt_bond_dim)
    for dens in dynamics(influance_matrix, phis):
        assert (jnp.abs(dens - dens.conj().T) < ACC).all()
        assert (jnp.linalg.eigvalsh(dens) > -ACC).all()
        assert jnp.abs(jnp.trace(dens) - 1.) < ACC
    subkeys = split(subkeys[-1], time_steps)
    phis = [random_unitary_channel(2, subkey) for subkey in subkeys]
    phi = phis[-1]
    dens_true = jnp.array([1, 0, 0, 0], dtype=jnp.complex64)
    swap_and_phi = swap_and_phi_im(time_steps, phi)
    for i, dens in enumerate(dynamics(swap_and_phi, phis)[1:]):
        if i % 2 == 1:
            dens_true = jnp.tensordot(phis[time_steps - i], dens_true, axes=1)
            dens_true = jnp.tensordot(phi, dens_true, axes=1)
            assert (jnp.abs(dens.reshape((-1,)) - dens_true) < ACC).all()
        else:
            assert (jnp.abs(dens.reshape((-1,)) - 0.5 * jnp.eye(2).reshape((-1,))) < ACC).all()


@pytest.mark.parametrize("subkey", split(KEY, 2))
@pytest.mark.parametrize("time_steps", [2, 5])
@pytest.mark.parametrize("local_choi_rank", [1, 3])
@pytest.mark.parametrize("sqrt_bond_dim", [1, 5])
def test_coupled_dynamics(
        subkey: KeyArray,
        time_steps: int,
        local_choi_rank: int,
        sqrt_bond_dim: int,
):
    """Tests coupled dynamics"""
    subkeys = split(subkey, 3)
    influance_matrix1 = random_im(subkeys[0], time_steps, local_choi_rank, sqrt_bond_dim)
    influance_matrix2 = random_im(subkeys[1], time_steps, local_choi_rank, sqrt_bond_dim)
    phi = random_unitary_channel(4, subkeys[2])
    phi = phi.reshape((2, 2, 2, 2, 2, 2, 2, 2))
    phi = phi.transpose((0, 2, 1, 3, 4, 6, 5, 7))
    phi = phi.reshape((4, 4, 4, 4))
    for dens in coupled_dynamics(influance_matrix1, influance_matrix2, phi):
        assert (jnp.abs(dens - dens.conj().T) < ACC).all()
        assert (jnp.linalg.eigvalsh(dens) > -ACC).all()
        assert jnp.abs(jnp.trace(dens) - 1.) < ACC
