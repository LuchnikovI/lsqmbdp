#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${script_dir}/utils.sh"

# ------------------------------------------------------------------------------------------

# Determine a base image (Cuda based or not)
if [[ ${USE_CUDA} == 1 ]]; then
    log INFO "Cuda is ON"
    base_image="nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-devel-ubuntu${UBUNTU_VERSION}"
    jax_install="--upgrade \"jax[cuda${CUDA_MAJOR_VERSION}_local]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
else
    log INFO "Cuda is OFF"
    base_image="ubuntu:${UBUNTU_VERSION}"
    jax_install="--upgrade \"jax[cpu]\""
fi

# ------------------------------------------------------------------------------------------

log INFO "Building an image..."
cat > "${IMAGE_NAME}.def" <<EOF

Bootstrap: docker
From: ${base_image}

%setup
    chmod +x "$(dirname ${script_dir})/src/entrypoint.sh"
    chmod +x "$(dirname ${script_dir})/src/get_config.py"
    chmod +x "$(dirname ${script_dir})/src/gen_samples.py"
    chmod +x "$(dirname ${script_dir})/src/random_im.py"
    chmod +x "$(dirname ${script_dir})/src/train_im.py"
    chmod +x "$(dirname ${script_dir})/src/plot_logs.py"
    chmod +x "$(dirname ${script_dir})/src/plot_coupled_spins_dynamics.py"
    chmod +x "$(dirname ${script_dir})/src/preproc.py"

%post
    DEBIAN_FRONTEND=noninteractive apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common curl
    DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa
    DEBIAN_FRONTEND=noninteractive apt-get update
    DEBIAN_FRONTEND=noninteractive apt install -y git python3.10 python3-pip
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
    python3.10 -m pip install --upgrade pip
    python3.10 -m pip install -U setuptools
    python3.10 -m pip install numpy pytest -U mypy scipy pylint hydra-core matplotlib \
    chex argparse h5py pyyaml git+https://github.com/LuchnikovI/qgoptax/ ${jax_install}

%runscript
    "$(dirname "${script_dir}")/src/entrypoint.sh \$@"

%help
    This is the nevironment for running of the lsqmbdp numerical experiments

EOF

# ------------------------------------------------------------------------------------------

if singularity build -F "${IMAGE_NAME}.sif" "${IMAGE_NAME}.def";
then
    log INFO "Base image ${IMAGE_NAME} has been built"
    rm -f "${IMAGE_NAME}.def"
else
    log ERROR "Failed to build a base image ${IMAGE_NAME}"
    rm -f "${IMAGE_NAME}.def"
    exit 1
fi
