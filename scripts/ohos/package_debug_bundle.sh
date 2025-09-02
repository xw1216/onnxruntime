#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR=${ROOT_DIR}/build/ohos_riscv64/Debug
HELLO_BIN=${ROOT_DIR}/build/ohos_hello/ort_hello
OUT_DIR=${ROOT_DIR}/build/ohos_bundle
mkdir -p "${OUT_DIR}/lib"

if [[ ! -f ${BUILD_DIR}/libonnxruntime.so ]]; then
  echo "[WARN] libonnxruntime.so not found in Debug build dir: ${BUILD_DIR}" >&2
fi
cp -f ${BUILD_DIR}/libonnxruntime.so* "${OUT_DIR}/lib/" 2>/dev/null || true
cp -f "${HELLO_BIN}" "${OUT_DIR}/"

cat > ${OUT_DIR}/run.sh <<'EOF'
#!/usr/bin/env bash
export LD_LIBRARY_PATH="${PWD}/lib:${LD_LIBRARY_PATH}"
./ort_hello "$@"
EOF
chmod +x ${OUT_DIR}/run.sh

echo "Bundle prepared at ${OUT_DIR}"
