#!/usr/bin/env bash
set -euo pipefail

# Install ort_models assets (images/labels/models) into Linux (bianbu) install prefix.

PREFIX="installed/ort_bianbu_rv64_debug"
SRC_DIR="samples/ort_models"

if [ ! -d "$PREFIX" ]; then
  echo "[install-assets-linux] Install prefix $PREFIX not found. Run install tasks first." >&2
  exit 1
fi
if [ ! -d "$SRC_DIR" ]; then
  echo "[install-assets-linux] Source dir $SRC_DIR not found." >&2
  exit 1
fi

ASSETS_DIR="$PREFIX/assets"
mkdir -p "$ASSETS_DIR/images" "$ASSETS_DIR/labels" "$ASSETS_DIR/models"

shopt -s nullglob
cp -a "$SRC_DIR/images"/* "$ASSETS_DIR/images/" 2>/dev/null || true
cp -a "$SRC_DIR/labels"/* "$ASSETS_DIR/labels/" 2>/dev/null || true
cp -a "$SRC_DIR/models"/*.onnx "$ASSETS_DIR/models/" 2>/dev/null || true
shopt -u nullglob

echo "[install-assets-linux] Assets installed under $ASSETS_DIR (images/ labels/ models/)"
exit 0
