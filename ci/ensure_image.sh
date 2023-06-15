#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${script_dir}/utils.sh"

# ------------------------------------------------------------------------------------------

if [[ -f "${IMAGE_NAME}.sif" ]]; then
    :
else
    log INFO "${IMAGE_NAME}.sif image has not been found"
    . "${script_dir}/build_image.sh"
fi