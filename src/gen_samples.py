#!/usr/bin/python3.10

# pylint: skip-file

import logging
logging.getLogger("jax._src.xla_bridge").addFilter(lambda _: False)
from typing import Dict
import jax.numpy as jnp
from jax.random import split, PRNGKey, categorical
from jax import pmap, local_device_count, devices, device_put, Array
from sampler import gen_samples, im2sampler, log_prob_from_sampler, im_log_norm
from im import coarse_graining
import yaml # type: ignore
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from cli_utils import _hdf2im, _data2hdf

par_gen_samples = pmap(gen_samples,
                       in_axes=(0, None, None),
                       static_broadcasted_argnums=(2,))
par_log_prob_from_sampler = pmap(log_prob_from_sampler, in_axes=(None, 0))


@hydra.main(version_base=None, config_path="../experiments/configs")
def main(cfg: DictConfig):
    conf = list(cfg.items())[0][1]
    batch_size = int(conf.dataset_generation_params.batch_size)
    seed = int(conf.seed)
    kernels_per_time_step = [int(x) for x in conf.dataset_generation_params.kernels_per_time_step]
    output_dir=HydraConfig.get().run.dir
    main_cpu = devices("cpu")[0]
    local_devices_number = local_device_count()
    batches_number = int(conf.dataset_generation_params.batches_number)
    batches_per_coarse_graining = batches_number // len(kernels_per_time_step)
    device_samples_number = batch_size * batches_per_coarse_graining * len(kernels_per_time_step)
    total_samples_number = batch_size * batches_per_coarse_graining * len(kernels_per_time_step) * local_devices_number
    print(yaml.dump(
        {
            "machine_dependant_dataset_generation_params":
                {
                    "local_devices_number": local_devices_number,
                    "device_samples_number": device_samples_number,
                    "total_samples_number": total_samples_number,
                }
        },
        width=float("inf"),
    ))
    # --------------------------------------------------------------------------------
    influence_matrix = _hdf2im(output_dir)
    log_influence_matrix_norm = im_log_norm(influence_matrix)
    # --------------------------------------------------------------------------------
    dataset: Dict[int, Array] = {}
    log_prob_value = jnp.zeros((1,))
    for n in kernels_per_time_step:
        key = PRNGKey(seed)
        key, _ = split(key)
        key, _ = split(key)
        keys = split(key, batches_per_coarse_graining)
        coarse_grained_influence_matrix = coarse_graining(influence_matrix, n)
        sampler = im2sampler(coarse_grained_influence_matrix)
        time_steps = len(sampler)
        all_samples = device_put(jnp.zeros((0, time_steps), dtype=jnp.int8), device=main_cpu)
        for key in keys:
            subkeys = split(key, local_devices_number)
            samples = par_gen_samples(subkeys, sampler, batch_size)
            log_prob_value += par_log_prob_from_sampler(sampler, samples).sum()
            all_samples = jnp.concatenate([all_samples, device_put(samples, main_cpu).reshape((-1, time_steps))])
        assert all_samples.shape[0] == total_samples_number // len(kernels_per_time_step)
        dataset[n] = all_samples
    log_prob_value -= log_influence_matrix_norm
    dataset_layout = {}
    for n in dataset:
        dataset_layout[n] = list(dataset[n].shape)
    print(yaml.dump({ "dataset_layout": dataset_layout }))
    _data2hdf(dataset, output_dir)
    print(yaml.dump(
        {
            "loss_value_exact_model": float(-log_prob_value.real),
        },
        width=float("inf"),
    ))


if __name__ == '__main__':
    main()
