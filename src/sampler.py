"""Sampler"""

from typing import List
from jax import Array
import jax.numpy as jnp
from im import InfluenceMatrix
from constants import projs

Sampler = List[Array]

def im2sampler(
        influence_matrix: InfluenceMatrix,
) -> Sampler:
    """Turns influence matrix into a sampler.
    Args:
        influence_matrix: Influence matrix
    Return: a sampler"""

    def translate_ker(ker: Array) -> Array:
        left_bond, _, _, _, _, right_bond = ker.shape
        ker = jnp.tensordot(ker, projs, axes=[[1, 2], [3, 2]])
        ker = jnp.tensordot(ker, projs, axes=[[1, 2], [3, 2]])
        ker = ker.transpose((0, 2, 4, 3, 5, 1))
        ker = ker.reshape((left_bond, 4, 16, right_bond)) / 2.
        return ker
    sampler = [translate_ker(ker) for ker in influence_matrix]
    return sampler
