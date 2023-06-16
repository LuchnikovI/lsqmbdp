#!/usr/bin/python3.10

# pylint: skip-file
import os
import sys
import h5py # type: ignore
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../experiments/configs")
def main(cfg: DictConfig):
    conf = list(cfg.items())[0][1]
    im_name = conf.im_name
    output_dir = HydraConfig.get().run.dir
    src_im_dir = os.path.dirname(os.path.realpath(im_name))
    im_dst = h5py.File(output_dir + "/im_exact", 'w')
    im_src = h5py.File(src_im_dir + "/experiments/ims/" + im_name)
    im_dst_group = im_dst.create_group("im")
    for id in im_src.keys():
        ker = np.array(im_src[id])
        lb, _, rb = ker.shape
        ker = ker.reshape((lb, 2, 2, 2, 2, rb))
        im_dst_group[id] = ker

if __name__ == '__main__':
    main()