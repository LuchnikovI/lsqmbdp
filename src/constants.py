"""Constants used across the project."""

import jax.numpy as jnp

# Pauli matrices
sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
sy = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
s0 = jnp.eye(2, dtype=jnp.complex64)

# Tensor with all observables (Pauli matrices)
obs = jnp.concatenate([sx[jnp.newaxis], sy[jnp.newaxis], sz[jnp.newaxis]], axis=0)

# A tensor containing projectors for all observables and possible outcomes
#
#               ket       bra
#                 \       /
#                  2     3
#                   \   /
#                    ###
#     outcome ---0---###---1--- observable number
#                    ###
#
v = jnp.linalg.eigh(obs)[1].transpose((2, 0, 1))
v = v[..., jnp.newaxis] * v[:, :, jnp.newaxis].conj()
s0_proj = jnp.concatenate([jnp.eye(2)[jnp.newaxis, jnp.newaxis], jnp.zeros((1, 1, 2, 2))], axis=0)
projs = jnp.concatenate([v, s0_proj], axis=1)
