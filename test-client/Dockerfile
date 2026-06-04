# Build context: repo root
#   docker-compose build test-client
#
# The proto file lives in MACE/Protos/ so the build context must be the repo
# root to give both the application source and the proto file to the builder.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies including grpcio-tools (needed only at build time for
# stub generation, but small enough to keep in the runtime image).
COPY test-client/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt grpcio-tools

# Generate gRPC Python stubs from the proto file.
# Output lands in /app/generated/, matching the path expected by main.py.
COPY MACE/Protos/mace_inference.proto /tmp/
RUN mkdir -p generated && \
    python -m grpc_tools.protoc \
        -I /tmp \
        --python_out=generated \
        --grpc_python_out=generated \
        /tmp/mace_inference.proto

COPY test-client/ ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
