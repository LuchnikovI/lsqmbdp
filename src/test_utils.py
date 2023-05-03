"""This is tests for utilities."""

from jax.random import KeyArray, PRNGKey, split
import jax.numpy as jnp
import pytest
from utils import _gen_random_channel

KEY = PRNGKey(42)

@pytest.mark.parametrize("subkey", split(KEY, 2))
@pytest.mark.parametrize("local_choi_rank", [1, 2, 4])
@pytest.mark.parametrize("inp_dim,out_dim",
                         [
                             (1, 1),
                             (2, 2),
                             (4, 4),
                             (1, 4),
                         ])
def test_gen_random_channel(
        subkey: KeyArray,
        local_choi_rank: int,
        inp_dim: int,
        out_dim: int,
):
    "Test correctness of a random quantum channel generation"
    phi = _gen_random_channel(subkey, inp_dim, out_dim, local_choi_rank)
    phi = phi.reshape((out_dim, out_dim, inp_dim, inp_dim))
    tr_phi = phi.trace(axis1=0, axis2=1)
    assert (jnp.abs(tr_phi - jnp.eye(inp_dim)) < 1e-5).all()
    phi = phi.transpose((0, 2, 1, 3))
    phi = phi.reshape((out_dim * inp_dim, out_dim * inp_dim))
    assert (jnp.abs(phi - phi.T.conj()) < 1e-5).all()
    lmbd = jnp.linalg.eigvalsh(phi)
    assert (lmbd > -1e-5).all()
