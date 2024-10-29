#!/bin/bash

set -e

script_dir=$(dirname $0)
root_dir=$(realpath "$script_dir/..")

cmake -S "$root_dir" -B ./build -G "Unix Makefiles"