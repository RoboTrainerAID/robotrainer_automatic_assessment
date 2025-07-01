#!/bin/sh

CONTAINER_NAME=robotrainer_automatic_assessment
CONTAINER_TAG=anaconda3
CONDA_ENV=robotrainer_automatic_assessment

docker build \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg CONDA_ENV="${CONDA_ENV}" \
    -t ${CONTAINER_NAME}:${CONTAINER_TAG} \
    .

    # --no-cache \
    # --progress plain \
    # --build-arg CACHE_BUST="$(date +%s)" \