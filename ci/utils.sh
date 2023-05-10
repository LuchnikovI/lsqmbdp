#!/usr/bin/env bash

ci_utils_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

export USE_CUDA=${USE_CUDA:-0}
export LOG_LEVELS=${LOG_LEVELS:-'DEBUG INFO WARNING ERROR'}
export UBUNTU_VERSION=${UBUNTU_VERSION:-'20.04'}

# see https://github.com/google/jax for more information
export CUDA_VERSION=${CUDA_VERSION:-'11.8.0'}
export CUDNN_MAJOR_VERSION=${CUDNN_MAJOR_VERSION:-'8'}

export CUDA_MAJOR_VERSION="$(echo ${CUDA_VERSION} | grep -oP "[0-9]+" | head -1)"

# -------------------------------------------------------------------------------------------

log() {
    local severity=$1
    shift

    local ts=$(date "+%Y-%m-%d %H:%M:%S%z")

    # See https://stackoverflow.com/a/46564084
    if [[ ! " ${LOG_LEVELS} " =~ .*\ ${severity}\ .* ]] ; then
        log ERROR "Unexpected severity '${severity}', must be one of: ${LOG_LEVELS}"
        severity=ERROR
    fi

    # See https://stackoverflow.com/a/29040711 and https://unix.stackexchange.com/a/134219
    local module=$(caller | awk '
        function basename(file, a, n) {
            n = split(file, a, "/")
            return a[n]
        }
        { printf("%s:%s\n", basename($2), $1) }')

    case "${severity}" in
        ERROR)
            color_start='\033[0;31m' # Red
            ;;
        WARNING)
            color_start='\033[1;33m' # Yellow
            ;;
        INFO)
            color_start='\033[1;32m' # Light Green
            ;;
        DEBUG)
            color_start='\033[0;34m' # Blue
            ;;
    esac
    color_end='\033[0m'

    printf "${ts} ${color_start}${severity}${color_end} [${module}]: ${color_start}$*${color_end}\n" >&2
}
