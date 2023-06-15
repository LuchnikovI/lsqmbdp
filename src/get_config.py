#!/usr/bin/python3.10

# pylint: skip-file

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../experiments/configs")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()
