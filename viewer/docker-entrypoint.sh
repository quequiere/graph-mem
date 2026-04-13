#!/bin/sh
set -e

# Replace ${VAR} placeholders in index.html template with environment variables.
# Only substitute these 3 variables — avoids clobbering JS template literals.
envsubst '${NEO4J_HTTP_URL} ${NEO4J_USER} ${NEO4J_PASSWORD}' \
  < /usr/share/nginx/html/index.html.template \
  > /usr/share/nginx/html/index.html

exec "$@"
