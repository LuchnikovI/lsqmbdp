"Tests properties of constants"

import jax.numpy as jnp
from constants import inv_povm, povm

ACC = 1e-5

def test_povm():
    """Tests properties of the tetrahedral povm."""

    povm_dot = jnp.tensordot(povm, inv_povm, axes=2)
    assert jnp.linalg.norm(povm_dot - jnp.eye(4)) < ACC
