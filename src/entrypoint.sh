#!/usr/bin/env bash

export JAX_ENABLE_X64=True

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

get_help() {
cat << EOF
Usage:
  --test:                      runs tests;
  --typecheck:                 runs static code analysis;
  --lint:                      runs linter;
  --help:                      drops this message;
  gen_rand_im:                 generates a random IM and saves it (it is a hydra cli app "https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/");
  gen_samples:                 generates samples from IM (it is a hydra cli app "https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/");
  train_im:                    trains IM on saved samples (it is a hydra cli app "https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/");
  get_config:                  prints experiment config (it is a hydra cli app "https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/");
  preproc:                     preprocess an externel influence matrix and save it in the output dir (it is a hydra cli app "https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/");
  plot_logs:                   make a plot from logs (use --help option to get usage)
  plot_coupled_spins_dynamics: make a plot of coupled spins dynamics (use --help option to get usage)
EOF
}

case $1 in

  --test)
        python3.10 -m pytest
    ;;
  --typecheck)
        python3.10 -m mypy --exclude /qgoptax/ "${script_dir}"
    ;;
  --lint)
        python3.10 -m pylint --ignore-paths="/lsqmbdp/src/qgoptax" "${script_dir}"
    ;;
  --help)
        get_help
    ;;
  random_im)
        shift
        "${script_dir}/random_im.py" "$@"
    ;;
  gen_samples)
        shift
        "${script_dir}/gen_samples.py" "$@"
    ;;
  train_im)
        shift
        "${script_dir}/train_im.py" "$@"
    ;;
  get_config)
        shift
        "${script_dir}/get_config.py" "$@"
    ;;
  plot_logs)
        shift
        "${script_dir}/plot_logs.py" "$@"
    ;;
  plot_coupled_spins_dynamics)
        shift
        "${script_dir}/plot_coupled_spins_dynamics.py" "$@"
    ;;
  preproc)
        shift
        "${script_dir}/preproc.py" "$@"
    ;;
  *)
        echo "Unknown option: '$1'"
        get_help
        exit 1
    ;;

esac
