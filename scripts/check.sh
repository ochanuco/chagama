#!/usr/bin/env bash
# CI と production の preflight で共通の check。クライアントの依存を入れ、テストと配信 identity の契約を確かめる
set -euo pipefail
uv venv .venv
uv pip install --python .venv/bin/python -e . pillow numpy opencv-python scipy pytest onnxruntime websockets
PYTHONPATH=scripts .venv/bin/pytest -q
.venv/bin/python scripts/delivery_check.py
