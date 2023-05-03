#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

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
  *)
        echo "Unknown option: '$1'"
        echo "Usage: $(basename $0) (--test|--typecheck|--lint)"
        echo "More options will be implemented later"
        exit 1
    ;;
esac
