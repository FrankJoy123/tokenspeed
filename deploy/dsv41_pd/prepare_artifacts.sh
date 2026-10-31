#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
run "$VENV/bin/python" "$DEPLOY_DIR/prepare_artifacts.py"
