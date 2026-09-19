# Stage 1: build the React/Vue SPA
FROM node:20-alpine AS ui-build
WORKDIR /app
COPY src/OperatorConsole/package*.json ./
RUN npm ci --prefer-offline
COPY src/OperatorConsole/ ./
RUN npm run build

# Stage 2: serve with nginx
FROM nginx:alpine AS runtime
COPY --from=ui-build /app/dist /usr/share/nginx/html
# nginx.conf is injected via ConfigMap in k8s; this is used for local docker-compose
COPY src/OperatorConsole/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
