LSQMBDP stands for large scale quantum many body data processing

## Prerequisites

One needs to have a Singularity(Apptainer) container system installed on your computer (HPC cluster).

## Quick start

To run an experiment with either random or external influence matrix:
1. create your experiment config in either `./experiments/configs/random_im` or `./experiments/configs/physical_im` directory, one can use `default.yaml` config as a template;
2. if you run an experiment with an external influence matrix, put the matrix inside `./experiments/ims` directory, and reflect its name in the corresponding field in the config;
3. run an experiment script `./experiments/random_im.sh --config <config_name>.yaml` or `./experiments/physical_im.sh --config <config_name>.yaml`, to utilize all nvidia gpus of the system set the following env. variable `USE_CUDA=1`

If you run a script for the first time, it automatically creates a singularity image with all the dependencies and runs all the experiments inside this environment.
Results of an experiment are stored in `./experiments/output/<experiment_type>/<config_name>/<timestamp>`. it contains learning process logs (`logs.yaml`), exact and trained influence matrices (`im_exact`, `im_trained`), generated dataset of measurement outcomes (`im_data`) and some additional information dropped by hydra framework.

## Cli interface
Experiment scripts use a cli tool `./ci/runner.sh` that covers everything. To get a help message of how to use the cli tool, run `./ci/runner.sh --help`

## How to plot some results
1. to plot learning curves use `./ci/runner plot_logs`, to get a help message run `./ci/runner.sh plot_logs --help`;
2. to plot dynamics of coupled spins interacting with influence matrices use `./ci/runner.sh plot_coupled_spins_dynamics`, to get a help message run `./ci/runner.sh plot_coupled_spins_dynamics --help`.

## How to run tests
`./ci/run_ci.sh` or `USE_CUDA=1 ./ci/run_ci.sh`