# Stage 1: build
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY src/SensorGateway/*.csproj ./SensorGateway/
RUN dotnet restore SensorGateway/SensorGateway.csproj
COPY src/SensorGateway/ ./SensorGateway/
RUN dotnet publish SensorGateway/SensorGateway.csproj -c Release -o /app/publish

# Stage 2: runtime
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /app/publish .
EXPOSE 8080 8081 9090
ENTRYPOINT ["dotnet", "SensorGateway.dll"]
