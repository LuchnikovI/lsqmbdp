"""Influence matrix"""

from typing import List
from chex import dataclass
import jax.numpy as jnp
from jax.random import KeyArray, split
from jaxtyping import Array, Complex64
from utils import _gen_random_channel

@dataclass
class InfluenceMatrix:
    """Influence matrix class"""
    kers: List[Complex64[Array, "#bd 2 2 2 2 #bd"]]


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
    Return: random influence matrix"""

    def gen_ker(
            out_dim: int,
            inp_dim: int,
            subkey: KeyArray
    ) -> Complex64[Array, "_l 2 2 2 2 _r"]:
        ker = _gen_random_channel(subkey, 2 * inp_dim, 2 * out_dim, local_choi_rank)
        ker = ker.reshape((2, out_dim, 2, out_dim, 2, inp_dim, 2, inp_dim))
        ker = ker.transpose((1, 3, 0, 2, 4, 6, 5, 7))
        ker = ker.reshape((out_dim * out_dim, 2, 2, 2, 2, inp_dim * inp_dim))
        return ker
    subkeys = split(key, time_steps)
    out_dims = time_steps * [sqrt_bond_dim]
    inp_dims = (time_steps - 1) * [sqrt_bond_dim] + [1]
    kers = [gen_ker(out_dim, inp_dim, subkey) for (out_dim, inp_dim, subkey)\
             in zip(out_dims, inp_dims, subkeys)]
    for ker in kers:
        print(ker.shape)
    last_ker =  kers[0]
    last_ker = last_ker.reshape(
        (sqrt_bond_dim, sqrt_bond_dim, 2, 2, 2, 2, -1)
    )
    last_ker = last_ker.trace(axis1=0, axis2=1)
    kers[0] = last_ker[jnp.newaxis]
    return InfluenceMatrix(kers = kers)


def im2phi(
        influance_matrix: InfluenceMatrix,
) -> Complex64[Array, "dim dim dim dim"]:
    """Transforms an influence matrix to a quantum channel.
    Args:
        im: influence matrix
    Return: quantum channel"""

    phi = jnp.ones((1, 1, 1, 1, 1), dtype=jnp.complex64)
    for ker in influance_matrix.kers:
        phi = jnp.tensordot(phi, ker, axes=1)
        phi = phi.transpose((0, 4, 1, 5, 2, 6, 3, 7, 8))
        new_dim = phi.shape[0] * phi.shape[1]
        phi = phi.reshape((new_dim, new_dim, new_dim, new_dim, -1))
    return phi[..., 0]
