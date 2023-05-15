#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# ------------------------------------------------------------------------------------------------
# This is a runner for a numerical experiment aimed on reconstruction of a random influence
# matrix from measurement outcomes in a random basis. Parameters of an experiment are set by
# environment variables inside a container.
# -------------------------------------------------------------------------------------------------
export NAME=${NAME:-"simple_random_im"}  # experiment name
export USE_CUDA=${USE_CUDA:-1}  # 1 if you want to use cuda, 0 if not

# Here one can modify parameters of the experiment
export ENVIRONMENT_DOCKER_ENV=${ENVIRONMENT_DOCKER_ENV:-"
--env LEARNING_RATE_IN=0.25
--env LEARNING_RATE_FINAL=0.0001
--env EPOCHS_NUMBER=300
--env SQ_BOND_DIM=7
--env SQ_BOND_DIM_TRAINING=8
--env TIME_STEPS=50
--env SAMPLES_NUMBER=1000
--env SAMPLES_NUMBER_TRAINING=10000
--env TOTAL_SAMPLES_NUMBER=1000000
--env LOCAL_CHOI_RANK=1
--env LOCAL_CHOI_RANK_TRAINING=4
--env SEED=42
"}
# -------------------------------------------------------------------------------------------------

if [[ $USE_CUDA == 1 ]]; then
    cuda_flag="cuda"
    run_flags="--nv"
else
    cuda_flag="cpu"
    run_flags=""
fi

results_dir=${script_dir}/${NAME}
sing_image=${script_dir}/${NAME}.simg
image_result_dir=/lsqmbdp/shared_dir
entrypoint_scipt=/lsqmbdp/src/entrypoint.sh
image_name="luchnikovi/lsqmbdp.${cuda_flag}:latest"

mkdir -p ${results_dir}

exec="singularity exec ${ENVIRONMENT_DOCKER_ENV} --bind ${results_dir}:${image_result_dir} ${run_flags} ${sing_image} . ${entrypoint_scipt}"
exec_get_params="${exec} --get_params"
exec_gen_rand_im="${exec} --gen_rand_im -n ${NAME}"
exec_gen_samples="${exec} --gen_samples -n ${NAME}"
exec_train_im="${exec} --train_im -n ${NAME}"


if singularity build ${sing_image}  docker://${image_name} > /dev/null; then
    echo "lsqmbd sigularity image has been built"
else
    echo "ERROR: Unable to build lsqmbd singularity image"
    exit 1
fi

# First, we print all the parameters of an experiment.
# Note, that you can modify any of them by setting the
# corresponding environment variable in the docker container
# https://stackoverflow.com/questions/30494050/how-do-i-pass-environment-variables-to-docker-containers
${exec_get_params}

# generates a random influence matrix
if ${exec_gen_rand_im}; then
    echo "Random IM is generated"
else
    echo "ERROR: unable to generate a random IM"
    exit 1
fi

# generates samples (measurement outcomes) from this matrix
if ${exec_gen_samples}; then
    echo "Measurement outcomes are generated"
else
    echo "ERROR: Unable to generate measurement outcomes"
    exit 1
fi

# trains an influence matrix
if ${exec_train_im}; then
    echo "Training is completed!"
else
    echo "ERROR: training is interrupted by an error"
    exit 1
fi
