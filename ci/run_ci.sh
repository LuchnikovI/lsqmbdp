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

docker run lsqmbdp.${cuda_tag}:${VERSION} --typecheck

log INFO "Running tests..."

docker run lsqmbdp.${cuda_tag}:${VERSION} --test

log INFO "Done!"