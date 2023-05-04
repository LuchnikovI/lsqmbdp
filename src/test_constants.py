"Tests properties of constants"

import jax.numpy as jnp
from constants import projs

ACC = 1e-5

def test_projs():
    """Tests properties of the projectors."""
    for proj in projs.transpose((1, 0, 2, 3)):
        assert (jnp.abs(proj[0] - proj[0].conj().T) < ACC).all()
        assert (jnp.abs(proj[1] - proj[1].conj().T) < ACC).all()
        assert (jnp.abs(proj[0] @ proj[0] - proj[0]) < ACC).all()
        assert (jnp.abs(proj[1] @ proj[1] - proj[1]) < ACC).all()
        assert (jnp.abs(proj[0] @ proj[1]) < ACC).all()
