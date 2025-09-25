##############################################################################
##                                 Base Image                               ##
##############################################################################
# Debian 12 (Bookworm) 
FROM continuumio/miniconda3:4.12.0
ENV TZ=Europe/Berlin
ENV TERM=xterm-256color
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

##############################################################################
##                                  User                                    ##
##############################################################################
# ARG USER=docker
# ARG PASSWORD=docker
# ARG UID=1000
# ARG GID=1000
# ENV USER=${USER}
# RUN groupadd -g ${GID} ${USER} && \
#     useradd -m -u ${UID} -g ${GID} -p "$(openssl passwd -1 ${PASSWORD})" --shell $(which bash) ${USER} -G sudo
# RUN echo "%sudo ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/sudogrp
# RUN usermod -a -G video ${USER}

##############################################################################
##                                 Global Dependecies                       ##
##############################################################################
# Install default packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    iputils-ping nano htop git sudo wget curl gedit gdb lsb-release bash-completion \
    build-essential swig python3-dev python3-setuptools python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Install custom dependencies
# RUN apt-get update && apt-get install --no-install-recommends -y \
#     <YOUR_PACKAGE> \
#     && rm -rf /var/lib/apt/lists/*

# RUN pip install \
#     <YOUR_PACKAGE>

COPY .bashrc /root/.bashrc

##############################################################################
##                                    Python-Pakete                                  ##
##############################################################################
RUN pip install numpy==1.21.6 matplotlib pandas scikit-learn scikit-optimize
RUN pip install auto-sklearn

##############################################################################
##                                 Autostart                                ##
##############################################################################

ARG CONDA_ENV
RUN sed -i "s|conda activate.*|conda activate /opt/conda/envs/${CONDA_ENV}|" /root/.bashrc

CMD ["bash"]
