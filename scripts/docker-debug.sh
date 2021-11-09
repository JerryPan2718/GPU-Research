#!/bin/bash
export command="${1:-bash}"
eval homedir="$(printf "~/%q" "$relativepath")"

set -xe

#    --secret id=netrc,src=$homedir/.netrc \
DOCKER_BUILDKIT=1 docker build \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    -t gptinference .

nvidia-docker run \
    --ipc=host --init --rm -it \
    -v $(pwd):/app \
    -w /app \
    -u $(id -u):$(id -g) \
    gptinference $command
