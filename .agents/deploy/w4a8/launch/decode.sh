#!/usr/bin/env bash
set -Eeuo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../launch/engine.sh" w4a8 decode "$@"
