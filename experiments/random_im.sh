#!/usr/bin/env bash

# -------------------------------------------------------------------
export USE_CUDA=${USE_CUDA:-1}  # 1 if you want to use cuda, 0 if not
export NAME=${NAME:-"simple_random_im"}  # experiment name
# -------------------------------------------------------------------

if [[ $USE_CUDA == 1 ]]; then
    cuda_flag="cuda"
    run_flags="--gpus all"
else
    cuda_flag="cpu"
    run_flags=""
fi

image_name="luchnikovi/lsqmbdp.${cuda_flag}:latest"
exec="docker run -it ${run_flags} -v im_experiments:/lsqmbdp/shared_dir ${image_name}"

# check docker
if docker --version > /dev/null; then
    echo "Docker is found"
else
    echo "ERROR: docker is not found, please install docker first"
    exit 1
fi

# create a volume for computation results
if docker volume create im_experiments > /dev/null; then
    echo "A volume for computation results has been created"
else
    echo "ERROR: unable to create a volume for computation results"
    exit 1
fi

# download the lates lsqmbdp image
if docker image inspect $image_name &> /dev/null; then
    echo "lsqmbdp image is found"
else
    echo "lsqmbd image is not found -> donwloading..."
    if docker pull $image_name > /dev/null; then
        echo "lsqmbd image has been downloaded"
    else
        echo "ERROR: Unable to download lsqmbd image"
        exit 1
    fi
fi

# first, we pring all the parameters of an experiment
# note, that you can modify any of them by setting the
# corresponding environment variable in the docker container
# https://stackoverflow.com/questions/30494050/how-do-i-pass-environment-variables-to-docker-containers
${exec} --get_params

# generates a random influence matrix
if ${exec} --gen_rand_im -n ${NAME}; then
    echo "Random IM is generated"
else
    echo "ERROR: unable to generate a random IM"
    exit 1
fi

# generates samples (measurement outcomes) from this matrix
if ${exec} --gen_samples -n ${NAME}; then
    echo "Measurement outcomes are generated"
else
    echo "ERROR: Unable to generate measurement outcomes"
    exit 1
fi

# trains an influence matrix
if ${exec} --train_im -n ${NAME}; then
    echo "Training is completed!"
else
    echo "ERROR: training is interrupted by an error"
    exit 1
fi