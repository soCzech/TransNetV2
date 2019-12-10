FROM tensorflow/tensorflow:devel-gpu-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    rsync \
    unzip \
    zip \
    zlib1g-dev \
    wget \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    vim

RUN pip3 --no-cache-dir install \
    Pillow \
    h5py \
    keras_applications \
    keras_preprocessing \
    matplotlib \
    mock \
    numpy \
    scipy \
    sklearn \
    pandas \
    tensorflow-gpu \
    tqdm \
    ffmpeg-python \
    pyyaml \
    opencv-python \
    opencv-contrib-python \
    shapely

RUN git clone https://github.com/soCzech/gin-config && cd gin-config && python3 setup.py install
