FROM node:20-alpine AS build
WORKDIR /ui
COPY src/ui/package*.json ./
RUN npm ci
COPY src/ui .
RUN npm run build

FROM nginx:alpine
COPY --from=build /ui/dist /usr/share/nginx/html
