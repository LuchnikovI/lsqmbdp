#!/usr/bin/python3.10
#  type: ignore

# pylint: skip-file

import datetime
import logging
logging.getLogger("jax._src.xla_bridge").addFilter(lambda _: False)
import h5py # type: ignore
import jax.numpy as jnp
from jax.random import split, PRNGKey, permutation
from jax import pmap, local_device_count, devices, device_put
import yaml # type: ignore
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from qgoptax.manifolds import StiefelManifold # type: ignore
from qgoptax.optimizers import RAdam # type: ignore
from im import (
    params2im,
    #random_slow_params,
    random_params_weak_decay,
    # random_unitary_channel,
)
from mpa import set_to_forward_canonical, mpa_log_dot
from cli_utils import (
    _hdf2data,
    _hdf2im,
    par_trace_dist,
    par_dynamics_prediction,
    _av_grad,
    _loss_and_grad,
    _learning_rate_update,
)

par_random_params_weak_decay = pmap(random_params_weak_decay, static_broadcasted_argnums=(1, 2))


@hydra.main(version_base=None, config_path="../experiments/configs")
def main(cfg: DictConfig):
    conf = list(cfg.items())[0][1]
    batch_size = int(conf.training_params.batch_size)
    sqrt_bond_dim = int(conf.training_params.sqrt_bond_dim)
    epochs_number = int(conf.training_params.epochs_number)
    decay_epochs_number = int(conf.training_params.decay_epochs_number)
    local_choi_rank = int(conf.training_params.local_choi_rank)
    learning_rate_in = float(conf.training_params.learning_rate_in)
    learning_rate_out = float(conf.training_params.learning_rate_final)
    decay_law = conf.training_params.decay_law
    test_trajectories_number = int(conf.training_params.test_trajectories_number)
    seed = int(conf.seed)
    output_dir=HydraConfig.get().run.dir
    local_devices_number = local_device_count()
    data_tail_shape = (-1, local_devices_number, batch_size)
    main_cpu = devices("cpu")[0]
    # ---------------------------------------------------------------------------------
    unknown_influence_matrix = _hdf2im(output_dir)
    set_to_forward_canonical(unknown_influence_matrix)
    time_steps = len(unknown_influence_matrix)
    data = device_put(_hdf2data(output_dir), main_cpu)
    data = data.reshape((*data_tail_shape, 2, time_steps))
    print(yaml.dump(
        {
            "machine_dependant_training_params":
                {
                    "local_devices_number": local_devices_number,
                    "batches_number": data.shape[0],
                    "test_trajectories_number": test_trajectories_number * local_devices_number,
                }
        },
        width=float("inf"),
    ))
    # ---------------------------------------------------------------------------------
    key = PRNGKey(seed)
    key, _ = split(key)
    key, _ = split(key)
    key, _ = split(key)
    _, subkey = split(key)
    subkeys = jnp.tile(subkey, (local_devices_number, 1))
    params = par_random_params_weak_decay(
        subkeys,
        local_choi_rank,
        sqrt_bond_dim,
    )
    stman = StiefelManifold()
    lr = learning_rate_in
    opt = RAdam(stman, lr)
    opt_state = opt.init(params)
    best_loss_val = jnp.finfo(jnp.float32).max

    # evaluates im before training
    found_influence_matrix = params2im([ker[0] for ker in params], time_steps, local_choi_rank)
    set_to_forward_canonical(found_influence_matrix)
    log_fidelity_half, _ = mpa_log_dot(unknown_influence_matrix, found_influence_matrix)
    fidelity = jnp.exp(2 * log_fidelity_half)
    subkeys = split(key, local_devices_number * test_trajectories_number).reshape((local_devices_number, test_trajectories_number, 2))
    density_matrices = par_dynamics_prediction(unknown_influence_matrix, found_influence_matrix, subkeys)
    mean_trace_dist = par_trace_dist(density_matrices)[0].mean()
    print(yaml.dump(
        {
            "initial_metrics":
                {
                    "cosin_sim": float(fidelity),
                    "mean_trace_dist": float(mean_trace_dist),
                }
        },
        width=float("inf"),
    ))
    # training loop
    for i in range(1, epochs_number + 1):
        opt = RAdam(stman, lr)
        av_loss_val = 0.
        for data_slice in data:
            loss_val, grads = _loss_and_grad(params, data_slice, local_choi_rank, batch_size)
            grads = _av_grad(grads)
            params, opt_state = opt.update(grads, opt_state, params)
            av_loss_val += loss_val.sum()
        key, subkey = split(key)
        data = permutation(subkey, data.reshape((-1, 2, time_steps)), False).reshape((*data_tail_shape, 2, time_steps))
        found_influence_matrix = params2im([ker[0] for ker in params], time_steps, local_choi_rank)
        density_matrices = par_dynamics_prediction(unknown_influence_matrix, found_influence_matrix, subkeys)
        hf_trained = h5py.File(output_dir + "/im_trained", 'a')
        if av_loss_val < best_loss_val:
            best_loss_val = av_loss_val
            try:
                del hf_trained["im"]
            except:
                pass
            im_group = hf_trained.create_group("im")
            for j, ker in enumerate(found_influence_matrix):
                im_group.create_dataset(str(j), data=ker)
        hf_trained.close()
        set_to_forward_canonical(found_influence_matrix)
        log_fidelity_half, _ = mpa_log_dot(unknown_influence_matrix, found_influence_matrix)
        fidelity = jnp.exp(2 * log_fidelity_half)
        subkeys = split(key, local_devices_number * test_trajectories_number).reshape((local_devices_number, test_trajectories_number, 2))
        mean_trace_dist = par_trace_dist(density_matrices)[0].mean()
        print(yaml.dump(
            {
                i:
                    {
                        "learning_rate": lr,
                        "loss_value": float(av_loss_val),
                        "cosin_sim": float(fidelity),
                        "mean_trace_dist": float(mean_trace_dist),
                        "timestamp": datetime.datetime.now()
                    }
            },
            width=float("inf"),
        ))
        lr = _learning_rate_update(
            lr,
            i,
            learning_rate_in,
            learning_rate_out,
            decay_epochs_number,
            decay_law,
        )

if __name__ == '__main__':
    main()
