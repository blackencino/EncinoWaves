#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
    print 'Run bash tools/setup.sh once to install the Python wheels.'
    exit 1
fi
exec .venv/bin/python -m encino_waves view "$@"
