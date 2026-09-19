# Stage 1: build
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY src/IncidentManager/*.csproj ./IncidentManager/
RUN dotnet restore IncidentManager/IncidentManager.csproj
COPY src/IncidentManager/ ./IncidentManager/
RUN dotnet publish IncidentManager/IncidentManager.csproj -c Release -o /app/publish

# Stage 2: runtime
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /app/publish .
EXPOSE 8080 9090
ENTRYPOINT ["dotnet", "IncidentManager.dll"]
