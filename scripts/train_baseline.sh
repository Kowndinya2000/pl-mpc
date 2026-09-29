#!/usr/bin/env bash
set -euo pipefail
release_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec python "$release_root/scripts/train_baseline.py" "$@"
