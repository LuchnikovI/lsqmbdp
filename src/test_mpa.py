"""MPA tests"""

from copy import deepcopy
from typing import List
import jax.numpy as jnp
import pytest
from jax.random import KeyArray
from jax.random import split, PRNGKey, categorical
from mpa import (
    mpa_gen_random,
    mpa2tensor,
    mpa_sum,
    set_to_forward_canonical,
    mpa_dot,
    mpa_eval,
)

KEY = PRNGKey(42)

ACC = 1e-5

@pytest.mark.parametrize("subkey", split(KEY, 2))
@pytest.mark.parametrize("shapes",
                         [
                             [[1]],
                             [[2, 1, 3]],
                             [[2, 1, 1], [3, 1, 2], [2, 2], [3, 3, 3]]
                         ])
@pytest.mark.parametrize("bond_dim", [1, 5])
def test_mpa_gen_random(
        subkey: KeyArray,
        shapes: List[List[int]],
        bond_dim: int,
):
    """Test random MPA generation."""

    mpa = mpa_gen_random(subkey, shapes, bond_dim)
    size = len(shapes)
    left_bonds = [1] + (size - 1) * [bond_dim]
    right_bonds = (size - 1) * [bond_dim] + [1]
    for ker, shape, left_bond, right_bond in zip(mpa, shapes, left_bonds, right_bonds):
        assert ker.shape == (left_bond,) + tuple(shape) + (right_bond,)


@pytest.mark.parametrize("subkey", split(KEY, 2))
@pytest.mark.parametrize("shapes",
                         [
                             [[1]],
                             [[2, 1, 3]],
                             [[2, 1, 1], [3, 1, 2], [2, 2], [3, 3, 3]]
                         ])
@pytest.mark.parametrize("bond_dim", [1, 5])
def test_mpa_sum(
        subkey: KeyArray,
        shapes: List[List[int]],
        bond_dim: int,
):
    """Tests sum function via turning mpa into a solid tensor."""
    mpa = mpa_gen_random(subkey, shapes, bond_dim)
    sum1 = mpa_sum(mpa)
    sum2 = mpa2tensor(mpa).sum()
    assert jnp.abs((sum1 - sum2) / (sum1 + sum2)) < ACC


@pytest.mark.parametrize("subkey", split(KEY, 2))
@pytest.mark.parametrize("shapes",
                         [
                             [[1]],
                             [[2, 1, 3]],
                             [[2, 1, 1], [3, 1, 2], [2, 2], [6, 2, 2], [2, 1, 1, 3], [2], [10]]
                         ])
@pytest.mark.parametrize("bond_dim", [1, 5])
def test_dot_and_forward_canonical(
        subkey: KeyArray,
        shapes: List[List[int]],
        bond_dim: int,
):
    """Test dot and forward canonical through each other."""

    mpa = mpa_gen_random(subkey, shapes, bond_dim)
    mpa_copy = deepcopy(mpa)
    dot_val = mpa_dot(mpa, mpa)
    assert (jnp.abs(dot_val.imag / dot_val) < ACC).all()
    log_norm = set_to_forward_canonical(mpa)
    assert (jnp.abs(jnp.log(dot_val) / 2 - log_norm) < ACC).all()
    dot_val = mpa_dot(mpa_copy, mpa)
    assert (jnp.abs(dot_val.imag / dot_val) < ACC).all()
    assert (jnp.abs(jnp.log(dot_val) - log_norm) < ACC).all()
    for ker in mpa:
        left_bond = ker.shape[0]
        right_bond = ker.shape[-1]
        ker = ker.reshape((left_bond, -1, right_bond))
        kerker = jnp.tensordot(ker, ker.conj(), axes=[[0, 1], [0, 1]])
        assert((jnp.abs(kerker - jnp.eye(kerker.shape[0])) < ACC).all())


@pytest.mark.parametrize("subkey", split(KEY, 2))
@pytest.mark.parametrize("shapes",
                         [
                             [[1]],
                             [[4]],
                             [[2], [3], [4], [3], [2], [1], [5]]
                         ])
@pytest.mark.parametrize("bond_dim", [1, 5])
def test_eval(
        subkey: KeyArray,
        shapes: List[List[int]],
        bond_dim: int,
):
    """Tests eval function."""

    mpa = mpa_gen_random(subkey, shapes, bond_dim)
    arr = mpa2tensor(mpa).reshape((-1,))
    _, subkey = split(subkey)
    arr_indices = categorical(subkey, jnp.ones(arr.size), shape=(10,))
    mpa_indices = jnp.array(jnp.unravel_index(arr_indices, [s[0] for s in shapes])).T
    arr_values = arr[arr_indices]
    mpa_values = mpa_eval(mpa, mpa_indices)
    assert (jnp.abs(mpa_values - arr_values) / jnp.abs(mpa_values + arr_values) < ACC).all()
