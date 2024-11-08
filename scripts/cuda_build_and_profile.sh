
#!/bin/bash

set -e

script_dir=$(dirname $0)
root_dir=$(realpath "$script_dir/../")

cmake --build "$root_dir/build"
cd "$root_dir/build/src/cuda"

nsys profile --trace cuda,nvtx --cuda-memory-usage true  --force-overwrite true  --output profile_run ./cuda_app

