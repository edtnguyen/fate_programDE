#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="${ROOT}/.tmp"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${TMP_DIR}"

export MPLCONFIGDIR="${TMP_DIR}"
export TMPDIR="${TMP_DIR}"

${PYTHON_BIN} "${ROOT}/test_environment.py"
${PYTHON_BIN} -m unittest discover -s "${ROOT}/tests"
