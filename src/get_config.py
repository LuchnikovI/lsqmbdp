#!/usr/bin/python3.10

# pylint: skip-file

import hydra
import yaml # type: ignore
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../experiments/configs")
def main(cfg: DictConfig) -> None:
    (experiment_type, conf) = list(cfg.items())[0]
    print(yaml.dump({"experiment_type": experiment_type}))
    print(OmegaConf.to_yaml(conf))

if __name__ == "__main__":
    main()
