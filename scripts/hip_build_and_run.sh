#!/bin/bash

set -e

script_dir=$(dirname $0)
root_dir=$(realpath "$script_dir/../")

cmake --build "$root_dir/build"
cd "$root_dir/build/src/hip"

./hip_app
