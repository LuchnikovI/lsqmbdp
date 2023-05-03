"Utilities."

from typing import List
from jax.random import KeyArray, normal
from jax import Array
import jax.numpy as jnp

def _random_normal_complex(
        subkey: KeyArray,
        shape: List[int],
) -> Array:
    val = normal(subkey, shape + [2,])
    val = val[..., 0] + 1j * val[..., 1]
    return val


def _gen_random_channel(
        subkey: KeyArray,
        inp_dim: int,
        output_dim: int,
        local_choi_rank: int,
) -> Array:
    iso = _random_normal_complex(subkey, [output_dim * local_choi_rank, inp_dim])
    iso, _ = jnp.linalg.qr(iso)
    iso = iso.reshape((local_choi_rank, output_dim, inp_dim))
    phi = jnp.tensordot(iso, iso.conj(), axes=[[0], [0]])
    phi = phi.transpose((0, 2, 1, 3))
    phi = phi.reshape((output_dim ** 2, inp_dim ** 2))
    return phi
