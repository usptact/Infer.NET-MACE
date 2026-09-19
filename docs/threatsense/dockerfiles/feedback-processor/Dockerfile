# Stage 1: build
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY src/FeedbackProcessor/*.csproj ./FeedbackProcessor/
RUN dotnet restore FeedbackProcessor/FeedbackProcessor.csproj
COPY src/FeedbackProcessor/ ./FeedbackProcessor/
RUN dotnet publish FeedbackProcessor/FeedbackProcessor.csproj -c Release -o /app/publish

# Stage 2: runtime
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /app/publish .
EXPOSE 8080 9090
ENTRYPOINT ["dotnet", "FeedbackProcessor.dll"]
