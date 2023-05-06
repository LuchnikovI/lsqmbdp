## How to build
For a CPU based version:

`./ci/ensure_image.sh`

for a GPU based version:

`USE_CUDA=1 ./ci/ensure_image.sh`

## How to test
`./ci/run_ci.sh`

## How to run benchmarks
For a CPU based version:

`docker run lsqmbdp.cpu:0.0.1 --bench`

If you desire to emulate multiple devices (here we took 2 just for example):

`docker run -e XLA_FLAGS="--xla_force_host_platform_device_count=2" lsqmbdp.cpu:0.0.1 --bench`

For a GPU based version (one needs to have Nvidia Container Toolkit, Nvidia drives and at least one Nvidia GPU):

`docker run --gpus all lsqmbdp.cuda:0.0.1 --bench`
