#!/usr/bin/python3.10

# pylint: skip-file

import os
from argparse import ArgumentParser
import numpy as np
from scipy.linalg import expm # type: ignore
import matplotlib.pyplot as plt # type: ignore
from im import coupled_dynamics
from constants import sx, sy, sz
from cli_utils import _hdf2im, _hdf2trained_im

#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "Helvetica"
#})

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--im_path1', '-p1',
        required=True,
        help="path to a directory with all the data for the influence matrix coupled with the first spin",
    )
    parser.add_argument(
        '--im_path2', '-p2',
        required=True,
        help="path to a directory with all the data for the influence matrix coupled with the second spin",
    )
    parser.add_argument(
        '--xx_int', '-x',
        default='0',
        help="xx interaction amplitude between spins"
    )
    parser.add_argument(
        '--yy_int', '-y',
        default='0',
        help="yy interaction amplitude between spins"
    )
    parser.add_argument(
        '--zz_int', '-z',
        default='1',
        help="zz interaction amplitude between spins"
    )
    args = parser.parse_args()
    im_exact1 = _hdf2im(args.im_path1)
    im_exact2 = _hdf2im(args.im_path2)
    im_trained1 = _hdf2trained_im(args.im_path1)
    im_trained2 = _hdf2trained_im(args.im_path2)
    xx = float(args.xx_int)
    yy = float(args.yy_int)
    zz = float(args.zz_int)
    h = xx * np.kron(sx, sx) + yy * np.kron(sy, sy) + zz * np.kron(sz, sz)
    u = expm(1j * 8 * h).reshape((2, 2, 2, 2))
    phi = np.tensordot(u, u.conj(), axes=0)
    phi = phi.transpose((0, 4, 1, 5, 2, 6, 3, 7)).reshape((4, 4, 4, 4))
    lmbds_gen = [np.linalg.eigvalsh(rho) for rho in coupled_dynamics(im_exact1, im_exact2, phi)]
    lmbds_trained = [np.linalg.eigvalsh(rho) for rho in coupled_dynamics(im_trained1, im_trained2, phi)]
    plt.plot([lmbds[0] for lmbds in lmbds_gen], 'b')
    plt.plot([lmbds[1] for lmbds in lmbds_gen], 'r')
    plt.plot([lmbds[2] for lmbds in lmbds_gen], 'k')
    plt.plot([lmbds[3] for lmbds in lmbds_gen], 'g')
    plt.plot([lmbds[0] for lmbds in lmbds_trained], 'b--')
    plt.plot([lmbds[1] for lmbds in lmbds_trained], 'r--')
    plt.plot([lmbds[2] for lmbds in lmbds_trained], 'k--')
    plt.plot([lmbds[3] for lmbds in lmbds_trained], 'g--')
    plt.ylabel("lmbd")
    plt.xlabel("time")
    plt.savefig(os.path.dirname(args.im_path1) + "/coupled_dynamics_plot.pdf")


if __name__ == '__main__':
    main()
