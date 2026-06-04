# Build context must be the repo root:
#   docker build -f dockerfiles/mace-inference/Dockerfile -t mace-inference .
#
# Local run (Development — DEBUG logging, pool=1):
#   docker run --rm -p 8080:8080 -p 9090:9090 \
#     -e ASPNETCORE_ENVIRONMENT=Development \
#     mace-inference
#
# Production run (pool size and log level from appsettings.json):
#   docker run --rm -p 8080:8080 -p 9090:9090 mace-inference
#
# Override individual settings without a config file:
#   docker run --rm -p 8080:8080 -p 9090:9090 \
#     -e INFERENCE__POOLSIZE=8 \
#     -e SERILOG__MINIMUMLEVEL__DEFAULT=Debug \
#     mace-inference
#
# Ports:
#   8080  gRPC service      (H2C — cleartext HTTP/2)
#   9090  Prometheus metrics (HTTP/1.1)

# ── Stage 1: build ───────────────────────────────────────────────────────────
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Restore as a separate layer so the NuGet cache survives source-only changes.
COPY MACE/MACE.csproj   ./MACE/
COPY MACE/Protos/        ./MACE/Protos/
RUN dotnet restore MACE/MACE.csproj

COPY MACE/ ./MACE/
RUN dotnet publish MACE/MACE.csproj -c Release -o /app/publish

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
# Use the debian-based aspnet image — NOT alpine.
# Infer.NET bundles Intel MKL native libraries that require glibc.
# musl libc (Alpine) is incompatible with MKL and will crash on startup.
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /app/publish .

EXPOSE 8080 9090

ENTRYPOINT ["dotnet", "MACE.dll"]
