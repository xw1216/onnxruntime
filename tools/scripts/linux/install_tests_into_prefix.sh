#!/usr/bin/env bash
set -euo pipefail

# Copy test executables and testdata from build tree into the install prefix
# for Linux (bianbu) target, mirroring OHOS helper.

PREFIX="installed/ort_bianbu_rv64_debug"
BUILD_DIR="build/bianbu_riscv64/Debug"

if [ ! -d "$PREFIX" ]; then
  echo "[install-tests-linux] Install prefix $PREFIX not found. Run install tasks first." >&2
  exit 1
fi
if [ ! -d "$BUILD_DIR" ]; then
  echo "[install-tests-linux] Build dir $BUILD_DIR not found." >&2
  exit 1
fi

mkdir -p "$PREFIX/bin"

# Align with OHOS list and add onnx_test_runner for convenience.
TEST_BINS=(
  onnxruntime_test_all
  onnxruntime_mlas_test
  onnxruntime_shared_lib_test
  onnxruntime_global_thread_pools_test
  onnxruntime_mlas_q4dq
  onnxruntime_perf_test
  onnx_test_runner
)

COPIED=()
for b in "${TEST_BINS[@]}"; do
  if [ -x "$BUILD_DIR/$b" ]; then
    cp -a "$BUILD_DIR/$b" "$PREFIX/bin/"
    COPIED+=("$b")
  fi
done

# testdata
if [ -d "$BUILD_DIR/testdata" ]; then
  mkdir -p "$PREFIX/testdata"
  cp -a "$BUILD_DIR/testdata"/* "$PREFIX/testdata/" 2>/dev/null || true
fi

# samples (some tests reference sample models / data)
if [ -d "$BUILD_DIR/samples" ]; then
  mkdir -p "$PREFIX/samples"
  cp -a "$BUILD_DIR/samples"/* "$PREFIX/samples/" 2>/dev/null || true
fi

echo "[install-tests-linux] Copied: ${COPIED[*]}"
echo "[install-tests-linux] testdata -> $PREFIX/testdata (if existed)"
echo "[install-tests-linux] samples  -> $PREFIX/samples (if existed)"
exit 0
