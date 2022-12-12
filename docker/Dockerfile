FROM nvcr.io/nvidia/l4t-pytorch:r32.4.4-pth1.6-py3

ARG DEBIAN_FRONTEND=noninteractive
ARG SRC_DIR="/src"

RUN mkdir -p ${SRC_DIR}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        unzip \
        zip \
        nano \
        pkg-config \
        python3-h5py \
        libssl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get autoremove -y --purge python3-pip

RUN cd ${SRC_DIR} && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py

RUN python3 -m pip install \
    tqdm \
    cython \
    pycocotools \
    traitlets \
    pytest \
    pyzmq \
    awsiotsdk

RUN cd ${SRC_DIR} && \
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt  && \
    python3 setup.py install --plugins

RUN cd ${SRC_DIR} && \
    git clone https://github.com/NVIDIA-AI-IOT/trt_pose && \
    cd trt_pose && \
    python3 setup.py install

RUN echo "Installing OpenCV4 with CUDA" && \
    sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf" && ldconfig && \
    apt-get update && apt-get install -y \
        libjpeg-dev libpng-dev libtiff-dev \
        libavcodec-dev libavformat-dev libswscale-dev \
        libgtk2.0-dev libcanberra-gtk* \
        python3-dev python3-numpy python3-pip \
        libxvidcore-dev libx264-dev libgtk-3-dev \
        libtbb2 libtbb-dev libdc1394-22-dev \
        libv4l-dev v4l-utils \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libavresample-dev libvorbis-dev libxine2-dev \
        libfaac-dev libmp3lame-dev libtheora-dev \
        libopencore-amrnb-dev libopencore-amrwb-dev \
        libopenblas-dev libatlas-base-dev libblas-dev \
        liblapack-dev libeigen3-dev gfortran \
        libhdf5-dev protobuf-compiler \
        libprotobuf-dev libgoogle-glog-dev libgflags-dev \
        qt5-default && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN cd ${SRC_DIR} && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.0.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.0.zip && \
    unzip opencv.zip && unzip opencv_contrib.zip && \
    mv opencv-4.5.0 opencv && mv opencv_contrib-4.5.0 opencv_contrib && \
    rm opencv.zip && rm opencv_contrib.zip

RUN cd ${SRC_DIR}/opencv && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr \
        -D OPENCV_EXTRA_MODULES_PATH=${SRC_DIR}/opencv_contrib/modules \
        -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
        -D WITH_CUDA=ON \
        -D CUDA_ARCH_BIN=5.3 \
        -D CUDA_ARCH_PTX="" \
        -D WITH_CUDNN=ON \
        -D WITH_CUBLAS=ON \
        -D ENABLE_FAST_MATH=ON \
        -D CUDA_FAST_MATH=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D ENABLE_NEON=ON \
        -D WITH_QT=OFF \
        -D WITH_OPENMP=ON \
        -D WITH_OPENGL=ON \
        -D BUILD_TIFF=ON \
        -D WITH_FFMPEG=ON \
        -D WITH_GSTREAMER=ON \
        -D WITH_TBB=ON \
        -D BUILD_TBB=ON \
        -D BUILD_TESTS=OFF \
        -D WITH_EIGEN=ON \
        -D WITH_V4L=ON \
        -D WITH_LIBV4L=ON \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D BUILD_NEW_PYTHON_SUPPORT=ON \
        -D BUILD_opencv_python3=TRUE \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_EXAMPLES=OFF ..

RUN cd ${SRC_DIR}/opencv/build && \
    make -j4 && make install && \
    ldconfig && \
    make clean && apt-get update

RUN python3 -m pip install \
    modern_robotics \
    scipy \
    sklearn

RUN rm -rf ${SRC_DIR} && \
    echo "Dependencies built and installed"
