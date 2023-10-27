"""Constants used across the project."""

import jax.numpy as jnp

# Pauli matrices
sigmax = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
sigmay = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
sigmaz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
identity = jnp.eye(2)
sigma = jnp.concatenate(
    [
        sigmax[jnp.newaxis],
        sigmay[jnp.newaxis],
        sigmaz[jnp.newaxis],
    ],
    axis=0,
)

# Tetrahedral POVM
weights = jnp.array(
    [
                          0,               0,       1,
        2 * jnp.sqrt(2) / 3,               0,  -1 / 3,
           -jnp.sqrt(2) / 3,  jnp.sqrt(2 / 3), -1 / 3,
           -jnp.sqrt(2) / 3, -jnp.sqrt(2 / 3), -1 / 3,
    ]
).reshape((4, 3))
povm = 0.25 * (identity[jnp.newaxis] + jnp.tensordot(weights, sigma, axes=1))
inv_povm = jnp.linalg.inv(povm.reshape((4, 4))).reshape((2, 2, 4))
