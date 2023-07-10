#!/usr/bin/python3.10

# pylint: skip-file

import logging
logging.getLogger("jax._src.xla_bridge").addFilter(lambda _: False)
import jax.numpy as jnp
from jax.random import split, PRNGKey, categorical
from jax import pmap, local_device_count, devices, device_put
from sampler import gen_samples, im2sampler, log_prob_from_sampler, im_log_norm
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
    batches_number = int(conf.dataset_generation_params.batches_number)
    seed = int(conf.seed)
    output_dir=HydraConfig.get().run.dir
    main_cpu = devices("cpu")[0]
    local_devices_number = local_device_count()
    device_samples_number = batch_size * batches_number
    total_samples_number = batch_size * batches_number * local_devices_number
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
    sampler = im2sampler(influence_matrix)
    time_steps = len(sampler)
    # --------------------------------------------------------------------------------
    key = PRNGKey(seed)
    key, _ = split(key)
    key, _ = split(key)
    keys = split(key, batches_number)
    all_samples = device_put(jnp.zeros((0, time_steps), dtype=jnp.int8), device=main_cpu)
    log_prob_value = jnp.zeros((1,))
    for key in keys:
        subkeys = split(key, local_devices_number)
        samples = par_gen_samples(subkeys, sampler, batch_size)
        log_prob_value += par_log_prob_from_sampler(sampler, samples).sum()
        all_samples = jnp.concatenate([all_samples, device_put(samples, main_cpu).reshape((-1, time_steps))])
    log_prob_value -= log_influence_matrix_norm
    assert all_samples.shape[0] == total_samples_number
    _data2hdf(all_samples, output_dir)
    print(yaml.dump(
        {
            "loss_value_exact_model": float(-log_prob_value.real),
        },
        width=float("inf"),
    ))


if __name__ == '__main__':
    main()
