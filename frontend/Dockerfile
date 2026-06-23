FROM nginx:alpine

# Copy static frontend files
COPY index.html     /usr/share/nginx/html/index.html
COPY dashboard.html /usr/share/nginx/html/dashboard.html
COPY style.css      /usr/share/nginx/html/style.css
COPY app.js         /usr/share/nginx/html/app.js

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
