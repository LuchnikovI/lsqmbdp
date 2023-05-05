#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${script_dir}/utils.sh"

# Determine a base image (Cuda based or not)
if [[ ${USE_CUDA} == 1 ]]; then
    log INFO "Cuda is ON"
    base_image="nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-devel-ubuntu${UBUNTU_VERSION}"
    jax_install="pip install --upgrade pip && pip install --upgrade \"jax[cuda${CUDA_MAJOR_VERSION}_local]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    cuda_tag="cuda"
else
    log INFO "Cuda is OFF"
    base_image="ubuntu:${UBUNTU_VERSION}"
    jax_install="pip install --upgrade pip && pip install --upgrade \"jax[cpu]\""
    cuda_tag="cpu"
fi

log INFO "Building an image..."
docker build -t lsqmbdp.${cuda_tag}:${VERSION} -f - "${script_dir}/.." <<EOF

FROM ${base_image}

WORKDIR /lsqmbdp
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common curl&& \
    DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa && \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt install -y python3.10 python3-pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m pip install numpy
RUN python3.10 -m pip install pytest
RUN python3.10 -m pip install -U mypy
RUN python3.10 -m pip install pylint
RUN python3.10 -m pip install chex
RUN ${jax_install}
COPY ./src ./src
COPY ./ci/entrypoint.sh ./src/entrypoint.sh
RUN chmod +x ./src/entrypoint.sh

ENTRYPOINT [ "./src/entrypoint.sh" ]

EOF
