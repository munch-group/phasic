
ARG GITHUB_REPO=mbg-docker-exercises

ARG OS_TYPE=x86_64
ARG UBUNTU_VER=23.10
ARG CONDA_VER=23.10.0-1

ARG PY_VER=py38
# ARG PY_VER=py39
# ARG PY_VER=py310
# ARG PY_VER=py311
# ARG PY_VER=py312

# base image
FROM ubuntu:${UBUNTU_VER}

LABEL maintainer="Kasper Munch <kaspermunch@birc.au.dk>"

# send stdout and stderr straight to terminal
ENV PYTHONUNBUFFERED 1

# centally installed system packages. Always install curl and lib6
# (this is where you install any packages required for compilation 
# like build-essential libc6-dev libpthread-stubs0-dev)
RUN apt update \
    && apt install -y --no-upgrade libc6 curl \
    && apt remove --purge && rm -rf /var/lib/apt/lists/*

# # use the above args during building 
# https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG CONDA_VER
ARG OS_TYPE
ARG PY_VER
# Install miniconda to /miniconda
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${PY_VER}_${CONDA_VER}-Linux-${OS_TYPE}.sh" \
    && bash Miniconda3-${PY_VER}_${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b \
    && rm Miniconda3-${PY_VER}_${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda init

#############################################################

# install packages for Magdalena's exercise (REQUIRES python 3.8)
RUN conda update -n base conda \
    && conda install -n base -c conda-forge -c r -q -y \
        git zip mscorefonts graphviz gmp \
        jupyterlab jupyterlab-git nodejs ipynbname openssh nbdime\
        pandas pytables seaborn python-graphviz rpy2 \
        gsl r-foreach r-devtools r-rcppgsl r-essentials \
        r-viridis r-dagitty r-rsvg r-gganimate r-ggpubr r-roxygen2 \
    && conda clean -afy     

RUN chsh -s $(which bash) && nbdime config-git --enable --global


COPY docker-entrypoint.sh /bin/docker-entrypoint.sh
RUN chmod +x /bin/docker-entrypoint.sh
ENTRYPOINT ["/bin/docker-entrypoint.sh"]
CMD ["/bin/bash", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.terminado_settings=\"shell_command=['/bin/bash']\""]

# docker build --platform=linux/amd64 -t kaspermunch/phasetype:latest .

# docker run --rm -v $HOME/.ssh:/tmp/.ssh -v $(pwd):$(pwd) -w $(pwd) -i -t -p 8888:8888 

# docker push kaspermunch/jupyter-linux-amd64:latest