FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
        curl bzip2 software-properties-common pkg-config ca-certificates \
        cmake autoconf automake libtool flex sudo git tzdata openssh-server \
        libglib2.0-0 libxext6 libsm6 libxrender1 libreadline-dev \
        graphviz libgraphviz-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set timezone
ENV TZ Asia/Tokyo
RUN echo $TZ > /etc/timezone && rm /etc/localtime && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Install miniconda
ENV MINICONDA_VERSION 4.5.11
RUN curl -s -o miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash miniconda.sh -b -p /opt/miniconda && rm miniconda.sh
ENV PATH /opt/miniconda/bin:$PATH

# Create conda environments and install modules
ADD environment.yml /src/environment.yml
RUN conda env create -f /src/environment.yml
ADD requirements.txt /src/requirements.txt
RUN bash -c "source activate default && pip install -r /src/requirements.txt"
RUN bash -c "source activate default && \
        jupyter nbextension enable --py --sys-prefix widgetsnbextension && \
        jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
        python -m nltk.downloader -d /opt/miniconda/envs/default/share/nltk_data punkt"
ENV PYTHONHASHSEED 0

ADD . /src
WORKDIR /src

RUN bash -c "source activate default && python setup.py build_ext"
RUN bash -c "source activate default && python setup.py develop"
WORKDIR /work
ENTRYPOINT ["/work/entrypoint.sh"]
CMD /bin/bash
