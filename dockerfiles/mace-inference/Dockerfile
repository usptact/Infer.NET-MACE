# Stage 1: build
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY src/MaceInference/*.csproj ./MaceInference/
RUN dotnet restore MaceInference/MaceInference.csproj
COPY src/MaceInference/ ./MaceInference/
RUN dotnet publish MaceInference/MaceInference.csproj -c Release -o /app/publish

# Stage 2: runtime
# IMPORTANT: use debian-based image, NOT alpine.
# Infer.NET bundles Intel MKL native libraries that require glibc.
# musl libc (Alpine) is incompatible with MKL — the service will crash on startup.
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /app/publish .
EXPOSE 8080 9090
ENTRYPOINT ["dotnet", "MaceInference.dll"]
