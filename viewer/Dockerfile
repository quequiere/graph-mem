FROM cgr.dev/chainguard/nginx:latest-dev

USER root
RUN apk add --no-cache gettext
RUN chown -R nginx:nginx /usr/share/nginx/html
USER nginx

COPY index.html /usr/share/nginx/html/index.html.template
COPY docker-entrypoint.sh /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
