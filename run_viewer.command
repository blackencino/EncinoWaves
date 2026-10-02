#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
    print 'Run bash tools/setup.sh once to install the Python wheels.'
    exit 1
fi
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m encino_waves view "$@"
