#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
mkdir -p "$STATE_DIR/downloads" "$STATE_DIR/micromamba"
curl -fL --retry 2 --max-time 180 https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$STATE_DIR/downloads/micromamba.tar.bz2"
tar -xjf "$STATE_DIR/downloads/micromamba.tar.bz2" -C "$STATE_DIR/micromamba" bin/micromamba
"$STATE_DIR/micromamba/bin/micromamba" create -y --root-prefix "$STATE_DIR/mamba-root" -p "$STATE_DIR/compiler" --override-channels -c conda-forge 'gxx_linux-64=12'
"$STATE_DIR/compiler/bin/x86_64-conda-linux-gnu-g++" --version
