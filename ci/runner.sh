#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
entrypoint="$(dirname ${script_dir})/src/entrypoint.sh"

. "${script_dir}/utils.sh"

. "${script_dir}/ensure_image.sh"

if [[ ${USE_CUDA} == 1 ]]; then
    singularity exec --nv "${IMAGE_NAME}.sif" "${entrypoint}" "$@"
else
    singularity exec "${IMAGE_NAME}.sif" "${entrypoint}" "$@"
fi
