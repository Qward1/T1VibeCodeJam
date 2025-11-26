# Python runner image for interview platform
FROM python:3.11-slim

RUN useradd -m runner
WORKDIR /home/runner/app
COPY runner.py /home/runner/app/runner.py
USER runner
ENTRYPOINT ["python", "/home/runner/app/runner.py"]
