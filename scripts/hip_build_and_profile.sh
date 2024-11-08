
#!/bin/bash

set -e

script_dir=$(dirname $0)
root_dir=$(realpath "$script_dir/../")

cmake --build "$root_dir/build"
cd "$root_dir/build/src/hip"

trace_type="hip-trace --hip-activity"
trace_dir="out_traces"
trace_file="$trace_type"
rocprofv2 --$trace_type -d $trace_dir -o $trace_file --plugin perfetto ./hip_app
