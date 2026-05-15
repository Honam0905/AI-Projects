FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        coreutils \
        findutils \
        git \
        patch \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["sh", "-lc", "trap 'exit 0' TERM INT; while true; do sleep 3600; done"]
