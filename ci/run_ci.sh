#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${script_dir}/utils.sh"

. "${script_dir}/ensure_image.sh"

log INFO "Running typechecker..."

if [[ ${USE_CUDA} == 1 ]]; then
    cuda_tag="cuda"
else
    cuda_tag="cpu"
fi

exec=luchnikovi/lsqmbdp.${cuda_tag}:latest

if docker run ${exec} --typecheck; then
    log INFO Type checking OK
else
    log ERROR Type checking failed
    exit 1
fi

log INFO "Running tests..."

if docker run ${exec} --test; then
    log INFO Testing OK
else
    log ERROR Testing failed
    exit 1
fi

if docker run ${exec} --lint; then
    log INFO Linting OK
else
    log WARNING Linting failed
fi

log INFO "Done!"