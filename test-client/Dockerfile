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

COPY test-client/ ./

# Generate gRPC Python stubs from the proto file — runs after COPY so it
# always overwrites any stale stubs that may exist in test-client/generated/.
COPY MACE/Protos/mace_inference.proto /tmp/
RUN mkdir -p generated && \
    python -m grpc_tools.protoc \
        -I /tmp \
        --python_out=generated \
        --grpc_python_out=generated \
        /tmp/mace_inference.proto

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
