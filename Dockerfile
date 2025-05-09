# For Dev Container Environment
FROM tensorflow/tensorflow:2.19.0-gpu

# Fix library path to avoid tensorflow import error
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Remove cuDNN 8 manually. 
RUN apt-get remove -y libcudnn8 && apt-get autoremove -y

# Install cuDNN 9 manually. See following link.
# https://github.com/tensorflow/tensorflow/issues/62412
RUN apt-get update && apt-get -y install cudnn9-cuda-12

WORKDIR /

# Copy requirements.txt to dev container
COPY requirements.txt requirements.txt

# Install requirements 
RUN pip install --no-cache-dir -r requirements.txt

# Install Git and Vim
RUN apt-get update && apt-get install -y git && apt-get install -y vim