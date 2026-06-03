#!/usr/bin/env bash
# Generates Python gRPC stubs from the shared .proto file.
# Run once before starting the gateway, or after the proto changes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="$SCRIPT_DIR/../MACE/Protos"
OUT_DIR="$SCRIPT_DIR/generated"

mkdir -p "$OUT_DIR"
touch "$OUT_DIR/__init__.py"

python -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_DIR/mace_inference.proto"

echo "Stubs generated in $OUT_DIR"
