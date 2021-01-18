FROM nvcr.io/nvidia/l4t-pytorch:r32.4.4-pth1.6-py3

ARG DEBIAN_FRONTEND=noninteractive
ARG SRC_DIR="/src"

RUN mkdir -p ${SRC_DIR}

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        unzip \
        zip \
        nano \
        python3-opencv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    tqdm \
    cython \
    pycocotools \
    traitlets \
    pytest \
    pyzmq

RUN cd ${SRC_DIR} && \
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt  && \
    python3 setup.py install --plugins

RUN cd ${SRC_DIR} && \
    git clone https://github.com/NVIDIA-AI-IOT/jetcam && \
    cd jetcam && \
    sed -i 's/self._gst_str(), cv2.CAP_GSTREAMER/self.capture_device/g' jetcam/usb_camera.py && \
    python3 setup.py install

RUN cd ${SRC_DIR} && \
    git clone https://github.com/NVIDIA-AI-IOT/trt_pose && \
    cd trt_pose && \
    python3 setup.py install

RUN rm -rf ${SRC_DIR} && \
    echo "Dependencies built and installed"
