#!/usr/bin/env bash
# chmod +x this script if not already executable.
# Integrated build + install (runtime + sample models) + copy tests for OHOS riscv64
# Combines tasks: ort-ohos-install-all-debug + ort-ohos-install-tests-into-prefix
#
# Default layout (relative to repo root):
#   build/ohos_riscv64/Debug            - runtime build dir (produced by build.py)
#   build/ohos_models                   - sample models build dir
#   installed/ort_ohos_riscv64_debug    - install prefix
#
# Usage:
#   tools/scripts/ohos/build_install_with_tests.sh [options]
# Options:
#   --config <Debug|Release>     (default: Debug)
#   --prefix <dir>               (default: installed/ort_ohos_riscv64_debug)
#   --jobs <N>                   (parallel build jobs, default: nproc or 8)
#   --skip-runtime               Skip building & installing runtime
#   --skip-models                Skip building & installing sample models
#   --skip-tests                 Skip copying tests & testdata
#   --only-tests                 Only copy tests (assumes runtime+models already installed)
#   --only-models                Only build+install models (skip runtime build/install)
#   --clean-models               Remove models build dir before configure
#   --force-reconfigure          Force reconfigure models (delete CMakeCache.txt)
#   --build-script-extra <args>  Extra args appended to build.py invocation (quoted as one argument)
#   -h|--help                    Show help
#
# Environment overrides:
#   PYTHON        (python3 executable, default: python3)
#   ORT_BUILD_ARGS   Extra args for build.py (space separated)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}" >/dev/null

CONFIG=Debug
PREFIX="installed/ort_ohos_riscv64_debug"
RUNTIME_BUILD_ROOT="build/ohos_riscv64"
MODELS_BUILD_DIR="build/ohos_models"
TOOLCHAIN_FILE="${REPO_ROOT}/cmake/ohos_riscv64.toolchain.cmake"
PYTHON_BIN=${PYTHON:-python3}
JOBS=${JOBS:-}
[ -z "${JOBS}" ] && JOBS=$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)

SKIP_RUNTIME=0
SKIP_MODELS=0
SKIP_TESTS=0
ONLY_TESTS=0
ONLY_MODELS=0
CLEAN_MODELS=0
FORCE_RECONF=0
BUILD_SCRIPT_EXTRA=""

usage(){ sed -n '1,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | grep -v '^set -euo' ; }

while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --prefix) PREFIX="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --skip-runtime) SKIP_RUNTIME=1; shift;;
    --skip-models) SKIP_MODELS=1; shift;;
    --skip-tests) SKIP_TESTS=1; shift;;
    --only-tests) ONLY_TESTS=1; SKIP_RUNTIME=1; SKIP_MODELS=1; shift;;
    --only-models) ONLY_MODELS=1; SKIP_RUNTIME=1; shift;;
    --clean-models) CLEAN_MODELS=1; shift;;
    --force-reconfigure) FORCE_RECONF=1; shift;;
    --build-script-extra) BUILD_SCRIPT_EXTRA="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[ERR] Unknown option: $1" >&2; usage; exit 2;;
  esac
done

if [ ${ONLY_MODELS} -eq 1 ] && [ ${ONLY_TESTS} -eq 1 ]; then
  echo "[ERR] --only-models and --only-tests are mutually exclusive" >&2
  exit 2
fi

log(){ echo "[build-install] $*"; }

#############################################
# 1. Build & install runtime
if [ ${SKIP_RUNTIME} -eq 0 ]; then
  log "Building runtime (config=${CONFIG})"
  ${PYTHON_BIN} tools/ci_build/build.py \
    --ohos --ohos_arch riscv64 --ohos_ndk_root tools/ohos_ndk \
    --build_dir "${RUNTIME_BUILD_ROOT}" \
    --config "${CONFIG}" --update --build --parallel "${JOBS}" \
    --cmake_extra_defines onnxruntime_BUILD_SHARED_LIB=ON ${ORT_BUILD_ARGS:-} ${BUILD_SCRIPT_EXTRA}
  log "Installing runtime to ${PREFIX}"
  cmake --install "${RUNTIME_BUILD_ROOT}/${CONFIG}" --prefix "${PREFIX}"
else
  log "Skipping runtime build/install"
fi

#############################################
# 2. Build & install models sample
if [ ${SKIP_MODELS} -eq 0 ]; then
  if [ ${CLEAN_MODELS} -eq 1 ]; then
    log "Cleaning models build dir ${MODELS_BUILD_DIR}"
    rm -rf "${MODELS_BUILD_DIR}"
  fi
  if [ ${FORCE_RECONF} -eq 1 ]; then
    rm -f "${MODELS_BUILD_DIR}/CMakeCache.txt"
  fi
  if [ ! -f "${MODELS_BUILD_DIR}/CMakeCache.txt" ]; then
    log "Configuring sample models project"
    cmake -S samples/ohos/ort_models -B "${MODELS_BUILD_DIR}" \
      -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
      -DORT_ROOT="${REPO_ROOT}" \
      -DCMAKE_BUILD_TYPE="${CONFIG}" \
      -DORT_BUILD_DIR="${REPO_ROOT}/${RUNTIME_BUILD_ROOT}/${CONFIG}"
  else
    log "Reusing existing sample models configuration"
  fi
  log "Building sample models (jobs=${JOBS})"
  cmake --build "${MODELS_BUILD_DIR}" -j "${JOBS}"
  log "Installing sample models to ${PREFIX}"
  cmake --install "${MODELS_BUILD_DIR}" --prefix "${PREFIX}"
else
  log "Skipping models build/install"
fi

#############################################
# 3. Copy tests & testdata
if [ ${SKIP_TESTS} -eq 0 ]; then
  log "Copying tests + testdata into install prefix"
  bash tools/scripts/ohos/install_tests_into_prefix.sh || {
    echo "[WARN] install_tests_into_prefix failed" >&2; exit 1; }
else
  log "Skipping tests copy"
fi

log "Done. Install prefix content roots:"
for d in bin lib assets testdata; do
  [ -e "${PREFIX}/$d" ] && echo "  - ${PREFIX}/$d" || true
done

exit 0
