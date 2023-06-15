#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "$(dirname $script_dir)/ci/utils.sh"

runner="$(dirname $script_dir)/ci/runner.sh"

# -------------------------------------------------------------------------------

get_help() {
cat << EOF
Usage:
    --config [name]: sets the name of a config without .yaml extension;
    --help           drops this message.
EOF
}

experiment() {
    local config_flag
    local output_dir
    config_flag="+random_im=$1"
    output_dir="${script_dir}/output/random_im/$1/$(date "+%Y-%m-%d_%H:%M:%S%z")"
    logs="${output_dir}/logs.yaml"
    mkdir -p ${output_dir}

    shift

    log INFO "Experiment config:"
    $runner get_config hydra.run.dir="${output_dir}" "${config_flag}" "${@}" | tee -a $logs
    check_exit_code $? "Unable to inspect a config"

    log INFO "Generating a random influence matrix..."
    $runner randandom_im hydra.run.dir="${output_dir}" "${config_flag}" "${@}" | tee -a $logs
    check_exit_code $? "Unable to inspect a config"

    log INFO "Generating a dataset..."
    $runner gen_samples hydra.run.dir="${output_dir}" "${config_flag}" "${@}" | tee -a $logs
    check_exit_code $? "Unable to generate a dataset"

    log INFO "Training..."
    $runner train_im hydra.run.dir="${output_dir}" "${config_flag}" "${@}" | tee -a $logs
    check_exit_code $? "Unable to train a model"
}

# -------------------------------------------------------------------------------

case $1 in
    --config)
        shift
        experiment "$@"
    ;;
    --help)
        get_help
    ;;
    *)
        echo "Unknown option $1"
        get_help
        exit 1
    ;;
esac
