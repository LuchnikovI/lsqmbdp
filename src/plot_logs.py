#!/usr/bin/python3.10

# pylint: skip-file

import os
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import yaml # type: ignore
from argparse import ArgumentParser

#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "Helvetica"
#})

def main():
    parser = ArgumentParser()
    parser.add_argument('--path', help = "path to the logs.yaml file (`./experiments/output/<experiment_type>/<config_name>/<timestamp>/logs.yaml`)", required = True)
    args = parser.parse_args()
    with open(args.path, "r") as log_file:
        try:
            log = yaml.safe_load(log_file)
        except yaml.YAMLError as exc:
            print(exc)
    epochs_number = int(log["training_params"]["epochs_number"])
    loss_value_exact_model = float(log["loss_value_exact_model"])
    initial_cosin_sim = float(log["initial_metrics"]["cosin_sim"])
    initial_mean_trace_dist = float(log["initial_metrics"]["mean_trace_dist"])
    cosin_sim = [initial_cosin_sim] + [float(log[i]["cosin_sim"]) for i in range(1, epochs_number + 1)]
    mean_trace_dist = [initial_mean_trace_dist] + [float(log[i]["mean_trace_dist"]) for i in range(1, epochs_number + 1)]
    loss_value = [float(log[i]["loss_value"]) for i in range(1, epochs_number + 1)]
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Epoch number')
    ax1.plot(list(range(epochs_number + 1)), 1 - np.array(cosin_sim), '-', color=color)
    ax1.plot(list(range(epochs_number + 1)), mean_trace_dist, ':', color=color)
    ax1.set_yscale('log')
    ax1.legend(["1 - Cosine similarity", "Av. prediction accuracy (trace dist.)"])
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_yscale("log")
    ax2.set_ylabel('|loss - exact_loss|', color=color)
    ax2.plot(list(range(1, epochs_number + 1)),
             np.abs(np.array(loss_value) - np.ones((epochs_number,)) * loss_value_exact_model),
             color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig(os.path.dirname(args.path) + "/logs_plot.pdf")


if __name__ == '__main__':
    main()