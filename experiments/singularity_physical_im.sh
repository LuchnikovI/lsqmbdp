#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# ------------------------------------------------------------------------------------------------
# This is a runner for a numerical experiment aimed on reconstruction of a physical influence
# matrix from measurement outcomes in a random basis. Parameters of an experiment are set by
# environment variables inside a container.
# -------------------------------------------------------------------------------------------------

export NAME=${NAME:-"J_0.1000_chi_256"}  # experiment name
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

iter_total_samples_number=(1000000)
iter_local_choi_rank_training=(2)
iter_sq_bond_dim_training=(8)

# a function running an experiment
experiment() {
    export ENVIRONMENT_DOCKER_ENV="
    --env SEED=42
    --env LEARNING_RATE_IN=0.25 \
    --env LEARNING_RATE_FINAL=0.23 \
    --env EPOCHS_NUMBER=4 \
    --env SAMPLES_NUMBER=1000 \
    --env SAMPLES_NUMBER_TRAINING=5000 \
    --env TEST_TRAJECTORIES_NUMBER=1000 \
    --env TOTAL_SAMPLES_NUMBER=$1 \
    --env TIME_STEPS=50 \
    --env LOCAL_CHOI_RANK=1 \
    --env LOCAL_CHOI_RANK_TRAINING=$2 \
    --env SQ_BOND_DIM=7 \
    --env SQ_BOND_DIM_TRAINING=$3 \
    "
    # -------------------------------------------------------------------------------------------------

    experimet_specific_name=$(cat <<EOF
${NAME}\
/total_samples_number_$1\
/local_choi_rank_training_$2\
/sq_bond_dim_training_$3
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
        echo "# lsqmbd sigularity image is found"
    else
        echo "# lsqmbd sigularity image has not been found, building..."
        if singularity pull ${simg_image}  docker://${image_name}; then
            echo "# lsqmbd sigularity image has been built"
        else
            echo "# ERROR: Unable to build lsqmbd singularity image"
            exit 1
        fi
    fi

    # First, we print all the parameters of an experiment.
    # Note, that you can modify any of them by setting the
    # corresponding environment variable in the docker container
    # https://stackoverflow.com/questions/30494050/how-do-i-pass-environment-variables-to-docker-containers
    echo "# Starting experiment with the following set of parameters:"
    exec_fn --get_params

    # copies an influence matrix
    if cp ${script_dir}/../physical_ims/${NAME}.h5 ${results_dir}/im_gen; then
        echo "# IM is copied"
    else
        echo "# ERROR: unable to copy IM"
    fi

    # generates samples (measurement outcomes) from this matrix
    if exec_fn --gen_samples -n ${experimet_specific_name}/im; then
        echo "# Measurement outcomes are generated"
    else
        echo "# ERROR: Unable to generate measurement outcomes"
    fi

    # trains an influence matrix
    if exec_fn --train_im -n ${experimet_specific_name}/im; then
        echo "// Training is completed!"
    else
        echo "// ERROR: training is interrupted by an error"
    fi
}

# --------------------------------------------------------------------------------------

for total_samples_number in "${iter_total_samples_number[@]}"; do
    for local_choi_rank_training in "${iter_local_choi_rank_training[@]}"; do
        for sq_bond_dim_training in "${iter_sq_bond_dim_training[@]}"; do
            experiment $total_samples_number \
                $local_choi_rank_training \
                $sq_bond_dim_training
        done
    done
done
