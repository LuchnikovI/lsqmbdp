#!/usr/bin/env python3

# pylint: skip-file

import sys
import h5py # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore


sx = np.array([[0, 1], [1, 0]], dtype=np.complex64)
sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
sz = np.array([[1, 0], [0, -1]], dtype=np.complex64)


def main(argv):
    hf_prediction = h5py.File(argv[1], 'a')
    dynamics_group = hf_prediction["dynamics"]
    exact = dynamics_group["exact"]
    predicted = dynamics_group["predicted"]
    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(np.tensordot(exact, sx, axes=[[3, 4], [1, 0]])[0, 0], 'b-')
    ax[0].plot(np.tensordot(predicted, sx, axes=[[3, 4], [1, 0]])[0, 0], 'b:')
    ax[0].set_xticks([], [])
    ax[0].set_ylabel(r'$\langle x\rangle$')
    ax[1].plot(np.tensordot(exact, sy, axes=[[3, 4], [1, 0]])[0, 0], 'b-')
    ax[1].plot(np.tensordot(predicted, sy, axes=[[3, 4], [1, 0]])[0, 0], 'b:')
    ax[1].set_xticks([], [])
    ax[1].set_ylabel(r'$\langle y\rangle$')
    ax[2].plot(np.tensordot(exact, sz, axes=[[3, 4], [1, 0]])[0, 0], 'b-')
    ax[2].plot(np.tensordot(predicted, sz, axes=[[3, 4], [1, 0]])[0, 0], 'b:')
    ax[2].set_ylabel(r'$\langle z\rangle$')
    ax[2].set_xlabel(r'${\rm Time}$')
    plt.show()


if __name__ == '__main__':
    main(sys.argv)
