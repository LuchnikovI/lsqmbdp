#!/usr/bin/python3.10

# pylint: skip-file

import logging
logging.getLogger("jax._src.xla_bridge").addFilter(lambda _: False)
from im import random_im
from jax.random import split, PRNGKey
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from cli_utils import _im2hdf


@hydra.main(version_base=None, config_path="../experiments/configs")
def main(cfg: DictConfig):
    conf = list(cfg.items())[0][1]
    time_steps = int(conf.generated_im_params.time_steps)
    sqrt_bond_dim = int(conf.generated_im_params.sqrt_bond_dim)
    local_choi_rank = int(conf.generated_im_params.local_choi_rank)
    seed = int(conf.seed)
    output_dir=HydraConfig.get().run.dir
    # -----------------------------------------------------------------------------
    key = PRNGKey(seed)
    _, subkey = split(key)
    influence_matrix = random_im(subkey, time_steps, local_choi_rank, sqrt_bond_dim)
    assert len(influence_matrix) == time_steps
    _im2hdf(influence_matrix, output_dir)

if __name__ == '__main__':
    main()
