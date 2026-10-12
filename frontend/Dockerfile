# Multi-stage build: compile the Vite/React app in a throwaway Node image,
# then ship only the static output in a tiny nginx image. The ~1GB of
# node_modules and build tooling never ends up in the final image -- it
# exists only in the "builder" stage and is discarded.

# ---- builder ----
FROM node:20-slim AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# VITE_API_BASE is baked into the static JS at build time (Vite env vars
# are compile-time, not runtime) -- override it with --build-arg when the
# API will live somewhere other than localhost:5001.
ARG VITE_API_BASE=http://localhost:5001/api
ENV VITE_API_BASE=${VITE_API_BASE}
RUN npm run build

# ---- final: static files served by nginx ----
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
