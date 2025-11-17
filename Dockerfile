##############################################################################
##                                 Base Image                               ##
##############################################################################
# Debian 12 (Bookworm) 
FROM continuumio/anaconda3:main
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
# Install packages via conda for better compatibility
RUN conda install -c conda-forge scikit-learn pandas numpy scipy matplotlib seaborn -y
RUN pip install scikit-optimize
# Auto-sklearn alternative - use FLAML which is more modern and maintained
RUN pip install "flaml[automl]"


##############################################################################
##                                 Autostart                                ##
##############################################################################

ARG CONDA_ENV
RUN sed -i "s|conda activate.*|conda activate /opt/conda/envs/${CONDA_ENV}|" /root/.bashrc

CMD ["bash"]
