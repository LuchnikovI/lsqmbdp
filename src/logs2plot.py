#!/usr/bin/env python3

# pylint: skip-file

import sys
import os
import matplotlib.pyplot as plt # type: ignore
from typing import List, Dict
from lark import Lark, Transformer # type: ignore
from dataclasses import dataclass
from textwrap import wrap

@dataclass
class Experiment:
    fidelity: List[float]
    learning_rate: List[float]
    loss_value: List[float]
    trace_dist: List[float]
    params: Dict[str, float]
    plot_label: str



class LogsTransformer(Transformer):
    def start(self, items):
        return items
    def comment(self, items):
        c = ""
        for word in items:
            c += " " + str(word)
        return c
    def param(self, key_value):
        k, v = key_value
        return str(k), float(v)
    def record(self, key_value):
        k, v = key_value
        return str(k), float(v)
    def records(self, items):
        r = {}
        for k, v in items:
            r[k] = v
        return r
    def training_end(self, items):
        pass

def main(argv):
    script_dir = os.path.dirname(argv[0])
    parser = Lark.open(script_dir + "/logs_grammar.lark")
    with open(argv[1], "r") as logs:
        logs_str = logs.read()
    logs_parsed = LogsTransformer().transform(parser.parse(logs_str, start="start"))
    experiments = []
    experiment = Experiment([], [], [], [], {}, "")
    for line in logs_parsed:
        if isinstance(line, tuple):
            experiment.params[line[0]] = line[1]
            experiment.plot_label += str(line[0]) + " = " + str(line[1]) + ", "
        if isinstance(line, dict):
            try:
                experiment.learning_rate.append(line["LR"])
                experiment.loss_value.append(line["Loss_value"])
            except:
                pass
            experiment.fidelity.append(line["Fidelity"])
            experiment.trace_dist.append(line["L1"])
        if line is None:
            experiments.append(experiment)
            experiment = Experiment([], [], [], [], {}, "")
    for experiment in experiments:
        fig, ax1 = plt.subplots()
        plt.title("\n".join(wrap(experiment.plot_label, 60)))
        color = 'tab:red'
        ax1.set_xlabel('Epoch number')
        ax1.plot(experiment.fidelity, '-', color=color)
        ax1.plot(experiment.trace_dist, '--', color=color)
        ax1.legend(["Cosine similarity", "Av. prediction accuracy (trace dist.)"])
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('loss value', color=color)
        ax2.plot(range(1, len(experiment.loss_value) + 1), experiment.loss_value, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    main(sys.argv)