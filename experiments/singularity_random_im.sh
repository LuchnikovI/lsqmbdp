#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# ------------------------------------------------------------------------------------------------
# This is a runner for a numerical experiment aimed on reconstruction of a random influence
# matrix from measurement outcomes in a random basis. Parameters of an experiment are set by
# environment variables inside a container.
# -------------------------------------------------------------------------------------------------

export NAME=${NAME:-"simple_random_im"}  # experiment name
export USE_CUDA=${USE_CUDA:-1}  # 1 if you want to use cuda, 0 if not

# device specific flags
if [[ $USE_CUDA == 1 ]]; then
    cuda_flag="cuda"
    run_flags="--nv"
else
    cuda_flag="cpu"
    run_flags=""
fi

simg_image=${script_dir}/lsqmbdp.${cuda_flag}.simg  # full name of the singularity image
image_name=luchnikovi/lsqmbdp.${cuda_flag}:latest  # docker image name for uploading from the registry
shared_dir=/lsqmbdp/shared_dir  # sharded dir inside the container

iter_total_samples_number=(100000 10000000)
iter_time_steps=(50 25 75)
iter_local_choi_rank=(1 2)
iter_local_choi_rank_training=(4 16)
iter_sq_bond_dim=(7)
iter_sq_bond_dim_training=(8 6 10)

# a function running an experiment
experiment() {
    export ENVIRONMENT_DOCKER_ENV=${ENVIRONMENT_DOCKER_ENV:-"
    --env SEED=42
    --env LEARNING_RATE_IN=0.25 \
    --env LEARNING_RATE_FINAL=0.001 \
    --env EPOCHS_NUMBER=350 \
    --env SAMPLES_NUMBER=1000 \
    --env SAMPLES_NUMBER_TRAINING=2500 \
    --env TOTAL_SAMPLES_NUMBER=$1 \
    --env TIME_STEPS=$2 \
    --env LOCAL_CHOI_RANK=$3 \
    --env LOCAL_CHOI_RANK_TRAINING=$4 \
    --env SQ_BOND_DIM=$5 \
    --env SQ_BOND_DIM_TRAINING=$6 \
    "}
    # -------------------------------------------------------------------------------------------------

    # a common part of all the paths specific for the experimen
    experimet_specific_name=$(cat <<EOF
${NAME}\
/total_samples_number_$1\
/time_steps_$2\
/local_choi_rank_$3\
/local_choi_rank_training_$4\
/sq_bond_dim_$5\
/sq_bond_dim_training_$6
EOF
)

    # a dir where results are placed
    results_dir=${script_dir}/${experimet_specific_name}

    mkdir -p ${results_dir}

    # a function running a container
    exec_fn() {
        singularity run \
            --no-home \
            --cleanenv \
            --bind ${script_dir}:${shared_dir} \
            ${run_flags} \
            ${ENVIRONMENT_DOCKER_ENV} \
            ${simg_image} "$@"
    }

    if [[ -f ${simg_image} ]]; then
        echo "lsqmbd sigularity image is found"
    else
        echo "lsqmbd sigularity image has not been found, building..."
        if singularity pull ${simg_image}  docker://${image_name}; then
            echo "lsqmbd sigularity image has been built"
        else
            echo "ERROR: Unable to build lsqmbd singularity image"
            exit 1
        fi
    fi

    # First, we print all the parameters of an experiment.
    # Note, that you can modify any of them by setting the
    # corresponding environment variable in the docker container
    # https://stackoverflow.com/questions/30494050/how-do-i-pass-environment-variables-to-docker-containers
    echo "Starting experiment with the following set of parameters:"
    exec_fn --get_params

    # generates a random influence matrix
    if exec_fn --gen_rand_im -n ${experimet_specific_name}/im; then
        echo "Random IM is generated"
    else
        echo "ERROR: unable to generate a random IM"
        exit 1
    fi

    # generates samples (measurement outcomes) from this matrix
    if exec_fn --gen_samples -n ${experimet_specific_name}/im; then
        echo "Measurement outcomes are generated"
    else
        echo "ERROR: Unable to generate measurement outcomes"
        exit 1
    fi

    # trains an influence matrix
    if exec_fn --train_im -n ${experimet_specific_name}/im; then
        echo "Training is completed!"
    else
        echo "ERROR: training is interrupted by an error"
        exit 1
    fi
}

# --------------------------------------------------------------------------------------

for total_samples_number in "${iter_total_samples_number[@]}"; do
    for time_steps in "${iter_time_steps[@]}"; do
        for local_choi_rank in "${iter_local_choi_rank[@]}"; do
            for local_choi_rank_training in "${iter_local_choi_rank_training[@]}"; do
                for sq_bond_dim in "${iter_sq_bond_dim[@]}"; do
                    for sq_bond_dim_training in "${iter_sq_bond_dim_training[@]}"; do
                        experiment $total_samples_number \
                            $time_steps \
                            $local_choi_rank \
                            $local_choi_rank_training \
                            $sq_bond_dim \
                            $sq_bond_dim_training
                    done
                done
            done
        done
    done
done
