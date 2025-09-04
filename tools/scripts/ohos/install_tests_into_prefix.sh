#!/usr/bin/env bash
set -euo pipefail

# Copy test executables and testdata from build tree into the install prefix
# so that the regular deploy script will include them automatically.

PREFIX="installed/ort_ohos_riscv64_debug"
BUILD_DIR="build/ohos_riscv64/Debug"

if [ ! -d "$PREFIX" ]; then
  echo "[install-tests] Install prefix $PREFIX not found. Run install tasks first." >&2
  exit 1
fi
if [ ! -d "$BUILD_DIR" ]; then
  echo "[install-tests] Build dir $BUILD_DIR not found." >&2
  exit 1
fi

mkdir -p "$PREFIX/bin"

TEST_BINS=(
  onnxruntime_test_all
  onnxruntime_mlas_test
  onnxruntime_shared_lib_test
  onnxruntime_global_thread_pools_test
  onnxruntime_mlas_q4dq
  onnxruntime_perf_test
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
  # Copy contents not the directory itself to keep path consistent
  cp -a "$BUILD_DIR/testdata"/* "$PREFIX/testdata/" 2>/dev/null || true
fi

# samples (some tests reference sample models / data)
if [ -d "$BUILD_DIR/samples" ]; then
  mkdir -p "$PREFIX/samples"
  cp -a "$BUILD_DIR/samples"/* "$PREFIX/samples/" 2>/dev/null || true
fi

echo "[install-tests] Copied: ${COPIED[*]}"
echo "[install-tests] testdata -> $PREFIX/testdata (if existed)"
echo "[install-tests] samples  -> $PREFIX/samples (if existed)"
exit 0
