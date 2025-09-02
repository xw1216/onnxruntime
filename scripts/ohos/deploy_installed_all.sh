#!/usr/bin/env bash
set -euo pipefail

PREFIX="installed/ort_ohos_riscv64_debug"
STAGE_DIR="build/ort_sync_stage"
TARGET_ARCH_DIR="tools/ohos_ndk/llvm/lib/riscv64-linux-ohos"
REMOTE_BASE="/data/local/tmp/ort"

if [ ! -d "$PREFIX" ]; then
  echo "[deploy] Installed prefix $PREFIX not found. Run install tasks first." >&2
  exit 1
fi

rm -rf "$STAGE_DIR" && mkdir -p "$STAGE_DIR/ort"
# Preserve symlinks and perms
cp -a "$PREFIX"/* "$STAGE_DIR/ort/"
# Ensure libc++_shared.so included
if [ -f "$TARGET_ARCH_DIR/libc++_shared.so" ]; then
  mkdir -p "$STAGE_DIR/ort/lib"
  cp -a "$TARGET_ARCH_DIR/libc++_shared.so" "$STAGE_DIR/ort/lib/"
fi

TAR_FILE="build/ort_sync_stage.tar.gz"
rm -f "$TAR_FILE"
# Create tar with top-level content only, not containing the parent staging directory name
( cd "$STAGE_DIR/ort" && tar -czf "../../ort_sync_stage.tar.gz" . )

# Push and extract on device (requires tar on device)
hdc file send "$TAR_FILE" /data/local/tmp/ort_sync_stage.tar.gz
hdc shell "set -e; rm -rf $REMOTE_BASE; mkdir -p $REMOTE_BASE; cd $REMOTE_BASE; tar -xzf /data/local/tmp/ort_sync_stage.tar.gz"

# Recreate symlink if needed (libonnxruntime.so.1 pointing to full version)
FULL_VER=$(hdc shell "ls -1 $REMOTE_BASE/lib/libonnxruntime.so.* 2>/dev/null" | tr -d '\r' | grep -E 'libonnxruntime.so.[0-9]+\.[0-9]+\.[0-9]+' | head -n1 | xargs basename 2>/dev/null || true)
if [ -n "$FULL_VER" ]; then
  # Extract major part after first strip (e.g. libonnxruntime.so.1.22.2 -> libonnxruntime.so.1)
  MAJOR_LINK=$(echo "$FULL_VER" | sed -E 's/(libonnxruntime.so.[0-9]+)\..*/\1/')
  if [ "$MAJOR_LINK" != "$FULL_VER" ]; then
    hdc shell "cd $REMOTE_BASE/lib && ln -sf $FULL_VER $MAJOR_LINK"
  fi
  # Ensure plain libonnxruntime.so points to major link if present
  hdc shell "cd $REMOTE_BASE/lib && ln -sf $MAJOR_LINK libonnxruntime.so" || true
fi

echo "[deploy] Synchronized $PREFIX to $REMOTE_BASE"
