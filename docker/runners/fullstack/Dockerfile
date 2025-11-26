# Fullstack runner image (Node.js baseline, extend if needed)
FROM node:20-alpine

RUN adduser -D runner
WORKDIR /home/runner/app
COPY runner.js /home/runner/app/runner.js
USER runner
ENTRYPOINT ["node", "/home/runner/app/runner.js"]
