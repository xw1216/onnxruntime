#!/usr/bin/env bash
set -euo pipefail

PREFIX="installed/ort_ohos_riscv64_debug"
TARGET_ARCH_DIR="tools/ohos_ndk/llvm/lib/riscv64-linux-ohos"
REMOTE_BASE="/data/local/tmp/ort"
TAR_FILE="build/ort_sync_stage.tar.gz"

INCLUDE_MODE="all" # all | only-bin-lib
SKIP_TESTDATA=0
SKIP_SAMPLES=0
SKIP_MODELS=0
PRESERVE_REMOTE=0   # if 1 do not wipe remote, just overlay (faster incremental)

usage(){ cat <<EOF
Usage: $0 [options]
  --only-bin-lib        Deploy only bin/ and lib/ (plus libc++_shared) (implies skip models/testdata/samples)
  --skip-testdata       Skip testdata directory
  --skip-samples        Skip samples directory
  --skip-models         Skip models directory (if exists)
  --overlay             Do not delete remote root, overlay update
  -h|--help             Show this help

Environment overrides:
  PREFIX (default: $PREFIX)
  REMOTE_BASE (default: $REMOTE_BASE)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --only-bin-lib) INCLUDE_MODE="only-bin-lib"; SKIP_TESTDATA=1; SKIP_SAMPLES=1; SKIP_MODELS=1; shift;;
    --skip-testdata) SKIP_TESTDATA=1; shift;;
    --skip-samples) SKIP_SAMPLES=1; shift;;
    --skip-models) SKIP_MODELS=1; shift;;
    --overlay) PRESERVE_REMOTE=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "[deploy] Unknown option: $1" >&2; usage; exit 2;;
  esac
done

if [ ! -d "$PREFIX" ]; then
  echo "[deploy] Installed prefix $PREFIX not found. Run install tasks first." >&2
  exit 1
fi

rm -f "$TAR_FILE"

# Build include list
declare -a INCLUDE_ITEMS
if [ "$INCLUDE_MODE" = "only-bin-lib" ]; then
  INCLUDE_ITEMS=(bin lib)
else
  # enumerate top-level entries to keep ordering stable
  for item in bin lib models testdata samples; do
    [ "$item" = "testdata" ] && [ $SKIP_TESTDATA -eq 1 ] && continue
    [ "$item" = "samples" ] && [ $SKIP_SAMPLES -eq 1 ] && continue
    [ "$item" = "models" ] && [ $SKIP_MODELS -eq 1 ] && continue
    if [ -e "$PREFIX/$item" ]; then
      INCLUDE_ITEMS+=("$item")
    fi
  done
  # also include anything else (licenses etc.) if doing full mode
  if [ ${#INCLUDE_ITEMS[@]} -gt 0 ] && [ "$INCLUDE_MODE" = "all" ]; then
    # Add other non-dir files at root (e.g. VERSION_NUMBER)
    while IFS= read -r f; do
      [ -n "$f" ] && INCLUDE_ITEMS+=("$f")
    done < <(find "$PREFIX" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null)
  fi
fi

# Ensure libc++_shared.so present (copy directly into install prefix lib if missing)
if [ -f "$TARGET_ARCH_DIR/libc++_shared.so" ] && [ ! -f "$PREFIX/lib/libc++_shared.so" ]; then
  mkdir -p "$PREFIX/lib"
  cp -a "$TARGET_ARCH_DIR/libc++_shared.so" "$PREFIX/lib/"
  # If lib directory not already scheduled include it
  if [[ ! " ${INCLUDE_ITEMS[*]} " =~ " lib " ]]; then
    INCLUDE_ITEMS+=(lib)
  fi
fi

if [ ${#INCLUDE_ITEMS[@]} -eq 0 ]; then
  echo "[deploy] Nothing to include. Abort." >&2
  exit 1
fi

echo "[deploy] Packaging items: ${INCLUDE_ITEMS[*]}"

# Create tar (paths relative). Use subshell to avoid polluting PWD.
(
  cd "${PREFIX}" || exit 1
  tar -czf "../../$TAR_FILE" "${INCLUDE_ITEMS[@]}"
)

SIZE=$(du -h "$TAR_FILE" | awk '{print $1}')
echo "[deploy] Tar size: $SIZE"

# Push and extract
hdc file send "$TAR_FILE" /data/local/tmp/ort_sync_stage.tar.gz

if [ $PRESERVE_REMOTE -eq 0 ]; then
  hdc shell "set -e; rm -rf $REMOTE_BASE; mkdir -p $REMOTE_BASE"
else
  hdc shell "mkdir -p $REMOTE_BASE"
fi

hdc shell "set -e; cd $REMOTE_BASE; tar -xzf /data/local/tmp/ort_sync_stage.tar.gz"

# Recreate library symlinks
FULL_VER=$(hdc shell "ls -1 $REMOTE_BASE/lib/libonnxruntime.so.* 2>/dev/null" | tr -d '\r' | grep -E 'libonnxruntime.so.[0-9]+\.[0-9]+\.[0-9]+' | head -n1 | xargs basename 2>/dev/null || true)
if [ -n "$FULL_VER" ]; then
  MAJOR_LINK=$(echo "$FULL_VER" | sed -E 's/(libonnxruntime.so.[0-9]+)\..*/\1/')
  if [ "$MAJOR_LINK" != "$FULL_VER" ]; then
    hdc shell "cd $REMOTE_BASE/lib && ln -sf $FULL_VER $MAJOR_LINK"
  fi
  hdc shell "cd $REMOTE_BASE/lib && ln -sf $MAJOR_LINK libonnxruntime.so" || true
fi

echo "[deploy] Deployed to $REMOTE_BASE (mode=$INCLUDE_MODE overlay=$PRESERVE_REMOTE)"
