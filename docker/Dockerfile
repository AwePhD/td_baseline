# ---------------------- BUILD MMDET/MMCV ----------------------
# Build the mmdet / PSTR python package with CUDA compiler
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel AS build

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC
RUN apt-get update && \
    apt-get install -y gcc vim ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /opt/conda/bin/conda /bin/conda

SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
RUN git clone https://github.com/JialeCao001/PSTR.git /pstr && \
    pip install --no-cache-dir --upgrade pip wheel && \
    MMCV_WITH_OPS=1 pip install --no-cache-dir /pstr/mmcv && \
    pip install --no-cache-dir /pstr

# ---------------------- TD-BASE DEPENDENCIES ----------------------
# Create the base environment with the smaller runtime image
# We must get the built package from last container
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime AS base

COPY --from=build /opt/conda/lib/python3.8/site-packages/mmcv /opt/conda/lib/python3.8/site-packages/mmcv
COPY --from=build /opt/conda/lib/python3.8/site-packages/mmdet /opt/conda/lib/python3.8/site-packages/mmdet
RUN apt-get update && apt-get install -y vim git gcc && apt-get clean && rm -rf /var/lib/apt/lists/*

SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
COPY ./base.requirements.txt .
RUN pip install --no-cache-dir -r ./base.requirements.txt
