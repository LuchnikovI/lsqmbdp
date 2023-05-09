#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

export SQ_BOND_DIM=${SQ_BOND_DIM:-"8"}
export TIME_STEPS=${TIME_STEPS:-"70"}
export SAMPLES_NUMBER=${SAMPLES_NUMBER:-"1000"}
export LOCAL_CHOI_RANK=${LOCAL_CHOI_RANK:-"2"}
export SEED=${SEED:-"42"}

case $1 in

  --test)
        python3.10 -m pytest
    ;;
  --typecheck)
        python3.10 -m mypy --exclude /qgoptax/ "${script_dir}/../src"
    ;;
  --lint)
        pylint --ignore-paths="${script_dir}/../src/qgoptax" "${script_dir}/../src"
    ;;
  --bench)
        "${script_dir}/../src/benchmarks.py"
    ;;
  --gen_rand_im)
        shift
        "${script_dir}/../src/random_im.py" "$@"
    ;;
  --train_im)
        shift
        "${script_dir}/../src/train_im.py" "$@"
    ;;
  *)
        echo "Unknown option: '$1'"
        echo "Usage: $(basename $0) (--test|--typecheck|--lint|--bench|--gen_rand_im [options]|--train_im [options])"
        echo "More options will be implemented later"
        exit 1
    ;;
  
esac
