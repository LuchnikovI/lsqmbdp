"""This is tests for influence matrix."""

from jax.random import KeyArray, PRNGKey, split
import jax.numpy as jnp
import pytest
from im import (
    random_im,
    im2phi,
    random_unitary_im,
    dynamics,
    random_unitary_channel,
)

KEY = PRNGKey(42)

ACC = 1e-4

@pytest.mark.parametrize("subkey", split(KEY, 2))
@pytest.mark.parametrize("time_steps", [1, 5])
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
@pytest.mark.parametrize("time_steps", [1, 5])
@pytest.mark.parametrize("local_choi_rank", [1, 3])
@pytest.mark.parametrize("sqrt_bond_dim", [1, 5])
def test_random_unitary_im(
        subkey: KeyArray,
        time_steps: int,
        local_choi_rank: int,
        sqrt_bond_dim: int,
):
    """Tests random_unitary_im function by building
    an explicit channel and testing its properties"""
    influance_matrix = random_unitary_im(subkey, time_steps, local_choi_rank, sqrt_bond_dim)
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
@pytest.mark.parametrize("time_steps", [1, 5])
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
    influance_matrix = random_unitary_im(subkey, time_steps, local_choi_rank, sqrt_bond_dim)
    for dens in dynamics(influance_matrix, phis):
        dens - dens.T.conj()
        assert (jnp.abs(dens - dens.conj().T) < ACC).all()
        assert (jnp.linalg.eigvalsh(dens) > -ACC).all()
        assert jnp.abs(jnp.trace(dens) - 1.) < ACC
