# Build context must be the repo root:
#   docker build -f dockerfiles/mace-inference/Dockerfile -t mace-inference .
#
# Run:
#   docker run --rm -p 8080:8080 -p 9090:9090 mace-inference
#
# Override any setting without a config file (double underscore = section separator):
#   docker run --rm -p 8080:8080 -p 9090:9090 \
#     -e Mace__NumWorkers=12 \
#     -e Mace__PoolSize=8 \
#     -e Mace__Retention=0.995 \
#     mace-inference
#
# Persist what the service learns about workers across restarts:
#   docker run --rm -p 8080:8080 -p 9090:9090 \
#     -v mace-beliefs:/var/lib/mace \
#     -e Mace__BeliefStorePath=/var/lib/mace/workers.txt \
#     mace-inference
#
# Ports:
#   8080  gRPC            (h2c — cleartext HTTP/2)
#   9090  metrics, health (HTTP/1.1: /metrics, /healthz, /readyz)
#
# The two ports are not interchangeable. gRPC over plaintext requires HTTP/2, which a Prometheus
# scrape and a kubelet probe cannot speak, so each protocol gets its own listener.
#
# Platform note: pinned to linux/amd64.
#   Infer.NET bundles Intel MKL x86-64 native libraries, so the image cannot run on a native ARM64
#   host regardless of where it was built. Pinning here also avoids a linux/arm64 protoc segfault
#   (exit 139) in Grpc.Tools when building on Apple Silicon.

# ── Stage 1: build ───────────────────────────────────────────────────────────
FROM --platform=linux/amd64 mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Restore as its own layer so the NuGet cache survives source-only changes.
# MACE.Service references MACE, so both project files are needed before restore.
COPY MACE/MACE.csproj ./MACE/
COPY MACE.Service/MACE.Service.csproj ./MACE.Service/
COPY MACE.Service/Protos/ ./MACE.Service/Protos/
RUN dotnet restore MACE.Service/MACE.Service.csproj

COPY MACE/ ./MACE/
COPY MACE.Service/ ./MACE.Service/
RUN dotnet publish MACE.Service/MACE.Service.csproj -c Release -o /app/publish

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
# Debian-based, not alpine: Infer.NET's bundled MKL requires glibc and will not
# load against musl.
FROM --platform=linux/amd64 mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime
WORKDIR /app
COPY --from=build /app/publish .

EXPOSE 8080 9090

ENTRYPOINT ["dotnet", "MACE.Service.dll"]
