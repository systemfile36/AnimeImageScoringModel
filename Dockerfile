# For Dev Container Environment
FROM tensorflow/tensorflow:2.17.0-gpu

# Fix library path to avoid tensorflow import error
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

WORKDIR /

# Copy requirements.txt to dev container
COPY requirements.txt requirements.txt

# Install requirements 
RUN pip install --no-cache-dir -r requirements.txt

# Install Git
RUN apt-get update && apt-get install -y git