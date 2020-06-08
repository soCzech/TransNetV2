FROM tensorflow/tensorflow:2.1.1-gpu

RUN pip3 --no-cache-dir install \
    Pillow \
    ffmpeg-python

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg

COPY setup.py /tmp
COPY inference /tmp/inference

RUN cd /tmp && python3 setup.py install && rm -r *
