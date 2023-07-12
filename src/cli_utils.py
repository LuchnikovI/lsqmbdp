from functools import partial
from typing import Tuple
import jax.numpy as jnp
import numpy as np
from jax import Array, pmap, vmap, value_and_grad
from jax.lax import pmean, psum
from jax.random import KeyArray, split
import h5py # type: ignore
from im import InfluenceMatrix, random_unitary_channel, dynamics, InfluenceMatrixParameters
from sampler import log_prob


def _hdf2im(output_dir: str) -> InfluenceMatrix:
    with h5py.File(output_dir +"/im_exact") as f:
        def idx2ker(idx: int):
            ker = jnp.array(f["im"][str(idx)])
            return ker
        kers_num = len(f["im"].values())
        influence_matrix = [idx2ker(idx) for idx in range(kers_num-1, -1, -1)]
    return influence_matrix


def _hdf2trained_im(output_dir: str) -> InfluenceMatrix:
    with h5py.File(output_dir +"/im_trained") as f:
        def idx2ker(idx: int):
            ker = jnp.array(f["im"][str(idx)])
            return ker
        kers_num = len(f["im"].values())
        influence_matrix = [idx2ker(idx) for idx in range(kers_num-1, -1, -1)]
    return influence_matrix


def _im2hdf(
        influence_matrix: InfluenceMatrix,
        output_dir: str,
):
    with h5py.File(output_dir + "/im_exact", 'w') as f:
        group = f.create_group("im")
        for i, ker in enumerate(reversed(influence_matrix)):
            group.create_dataset(str(i), data=ker)


def _data2hdf(
        data: Array,
        output_dit: str,
):
    with h5py.File(output_dit + "/im_data", "w") as f:
        f.create_dataset("data", data=data)


def _hdf2data(
        path: str,
):
    with h5py.File(path + "/im_data") as f:
        data = np.array(f["data"])
    return data


@partial(pmap, in_axes=0, axis_name='i')
@partial(vmap, in_axes=0)
def par_trace_dist(
        density_matrices: Tuple[Array, Array],
) -> Array:
    spec = jnp.linalg.eigvalsh(density_matrices[0] - density_matrices[1])
    return pmean(jnp.abs(spec).sum(-1).mean(), "i")


@partial(pmap, in_axes=(None, None, 0))
@partial(vmap, in_axes=(None, None, 0))
def par_dynamics_prediction(
        influence_matrix: InfluenceMatrix,
        trained_influence_matrix: InfluenceMatrix,
        subkey: KeyArray,
) -> Tuple[Array, Array]:
    subkeys = split(subkey, len(influence_matrix))
    phis = [random_unitary_channel(2, subkey) for subkey in subkeys]
    density_matrices = dynamics(influence_matrix, phis)
    predicted_density_matrices = dynamics(trained_influence_matrix, phis)
    assert len(density_matrices) == len(predicted_density_matrices)
    return jnp.array(density_matrices), jnp.array(predicted_density_matrices)


@partial(pmap, in_axes=(0, 0, None), static_broadcasted_argnums=2)
@value_and_grad
def _loss_and_grad(
        params: InfluenceMatrixParameters,
        data: Array,
        local_choi_rank: int,
) -> Array:
    return -log_prob(params, data, local_choi_rank)


@partial(pmap, axis_name='i')
def _av_grad(
        grads: InfluenceMatrixParameters,
) -> InfluenceMatrixParameters:
    return psum(grads, axis_name='i')


def _learning_rate_update(
        learning_rate: float,
        iter_number: int,
        learning_rate_in: float,
        learning_rate_out: float,
        decay_epoches_number: int,
        decay_law: str,
) -> float:
    if iter_number <= decay_epoches_number:
        match decay_law:
            case "exponential":
                decay_coeff = (learning_rate_out / learning_rate_in) ** (1 / decay_epoches_number)
                learning_rate = learning_rate * decay_coeff
            case "linear":
                decrement = (learning_rate_in - learning_rate_out) / decay_epoches_number
                learning_rate = learning_rate - decrement
        return learning_rate
    else:
        return learning_rate
