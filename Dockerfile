FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Python 3.9 설치
RUN apt-get update
RUN apt-get install -y python3.9 curl
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN rm get-pip.py

# 필수 라이브러리 설치
RUN apt-get install -y build-essential cmake git

# PyTorch 2.2 이상 버전 설치
RUN pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
