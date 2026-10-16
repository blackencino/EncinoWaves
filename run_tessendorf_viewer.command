#!/bin/zsh
# Start the native School viewer at the early Phillips/Tweak baseline.
set -eu
exec "$(dirname "$0")/run_school_viewer.command" --tessendorf "$@"
