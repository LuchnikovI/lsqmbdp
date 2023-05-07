#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

export SQ_BOND_DIM=${SQ_BOND_DIM:-"8"}
export TIME_STEPS=${TIME_STEPS:-"70"}
export SAMPLES_NUMBER=${SAMPLES_NUMBER:-"1000"}
export SEED=${SEED:-"42"}

case $1 in

  --test)
        python3.10 -m pytest
    ;;
  --typecheck)
        python3.10 -m mypy "${script_dir}/../src"
    ;;
  --lint)
        pylint "${script_dir}/../src"
    ;;
  --bench)
        "${script_dir}/../src/benchmarks.py"
    ;;
  *)
        echo "Unknown option: '$1'"
        echo "Usage: $(basename $0) (--test|--typecheck|--lint|--bench)"
        echo "More options will be implemented later"
        exit 1
    ;;
  
esac
