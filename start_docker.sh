#!/bin/sh

# Autostart command to run inside the container, default is bash
# Usage1: Modify ./autostart.sh file and add custom command there
# Usage2: Run from cli with ./start_docker "custom command"
COMMAND=${1:-bash}
CONTAINER_NAME=robotrainer_automatic_assessment
CONTAINER_TAG=anaconda3

# Check if the container is already running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} is already running. Attaching to it..."
    docker exec -it ${CONTAINER_NAME} ${COMMAND}
    exit 0
fi

# Ensure XAUTHORITY is set
export XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}

docker run \
    --name ${CONTAINER_NAME} \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -it \
    --net host \
    --rm \
    -e DISPLAY=${DISPLAY} \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=${XAUTHORITY} \
    -v $XAUTHORITY:$XAUTHORITY:rw \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $PWD/envs:/opt/conda/envs \
    -v $PWD/notebooks:/opt/notebooks \
    -v $PWD/src:/opt/src \
    -v $PWD/data:/opt/data \
    ${CONTAINER_NAME}:${CONTAINER_TAG} \
    /bin/bash -c "${COMMAND}"

    # --env-file .env \