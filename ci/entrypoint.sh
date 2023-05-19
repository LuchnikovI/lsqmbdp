#!/usr/bin/env bash

export LEARNING_RATE_IN=${LEARNING_RATE_IN:-"0.25"}
export LEARNING_RATE_FINAL=${LEARNING_RATE_FINAL:-"0.0001"}
export EPOCHS_NUMBER=${EPOCHS_NUMBER:-"300"}
export SQ_BOND_DIM=${SQ_BOND_DIM:-"7"}
export SQ_BOND_DIM_TRAINING=${SQ_BOND_DIM_TRAINING:-"8"}
export TIME_STEPS=${TIME_STEPS:-"50"}
export SAMPLES_NUMBER=${SAMPLES_NUMBER:-"1000"}
export SAMPLES_NUMBER_TRAINING=${SAMPLES_NUMBER_TRAINING:-"10000"}
export TOTAL_SAMPLES_NUMBER=${TOTAL_SAMPLES_NUMBER:-"1000000"}
export LOCAL_CHOI_RANK=${LOCAL_CHOI_RANK:-"1"}
export LOCAL_CHOI_RANK_TRAINING=${LOCAL_CHOI_RANK_TRAINING:-"4"}
export SEED=${SEED:-"42"}

get_all_parameters() {
cat << EOF
LEARNING_RATE_IN=${LEARNING_RATE_IN}
LEARNING_RATE_FINAL=${LEARNING_RATE_FINAL}
EPOCHS_NUMBER=${EPOCHS_NUMBER}
SQ_BOND_DIM=${SQ_BOND_DIM}
SQ_BOND_DIM_TRAINING=${SQ_BOND_DIM_TRAINING}
TIME_STEPS=${TIME_STEPS}
SAMPLES_NUMBER=${SAMPLES_NUMBER}
SAMPLES_NUMBER_TRAINING=${SAMPLES_NUMBER_TRAINING}
TOTAL_SAMPLES_NUMBER=${TOTAL_SAMPLES_NUMBER}
LOCAL_CHOI_RANK=${LOCAL_CHOI_RANK}
LOCAL_CHOI_RANK_TRAINING=${LOCAL_CHOI_RANK_TRAINING}
SEED=${SEED}
EOF
}

get_help() {
cat << EOF
Usage
--test:                  runs tests;
--typecheck:             runs static code analysis;
--lint:                  runs linter;
--bench:                 runs benchmarks;
--gen_rand_im [-n name]: generates a random IM and saves it, optionally accepts name of the experiment (default to "random_im");
--gen_samples [-n name]: generates samples from saved IM, optionally accepts name of the experiment (default to "random_im");
--train_im    [-n name]: trains IM on saved samples, optionally accepts name of the experiment (default to "random_im");
--get_params:            prints all the environment variables essential for a numerical experiment;
--help                   drops this message.
EOF
}

case $1 in

  --test)
        python3.10 -m pytest
    ;;
  --typecheck)
        python3.10 -m mypy --exclude /qgoptax/ "/lsqmbdp/src"
    ;;
  --lint)
        pylint --ignore-paths="/lsqmbdp/src/qgoptax" "/lsqmbdp/src"
    ;;
  --bench)
        "/lsqmbdp/src/benchmarks.py"
    ;;
  --gen_rand_im)
        shift
        "/lsqmbdp/src/random_im.py" "$@"
    ;;
  --gen_samples)
        shift
        "/lsqmbdp/src/gen_samples.py" "$@"
    ;;
  --train_im)
        shift
        "/lsqmbdp/src/train_im.py" "$@"
    ;;
  --get_params)
        get_all_parameters
    ;;
  --help)
        get_help
    ;;
  *)
        echo "Unknown option: '$1'"
        get_help
        exit 1
    ;;
  
esac
