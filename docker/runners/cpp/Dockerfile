# C++ runner image
FROM gcc:13

RUN useradd -m runner
WORKDIR /home/runner/app
COPY runner.sh /home/runner/app/runner.sh
RUN chmod +x /home/runner/app/runner.sh
USER runner
ENTRYPOINT ["/home/runner/app/runner.sh"]
