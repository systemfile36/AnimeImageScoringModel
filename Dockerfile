# For project environment
FROM tensorflow/tensorflow:2.17.0-gpu

# Fix library path to avoid tensorflow import error
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

WORKDIR /project

# Copy all file in project to WORKDIR/
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=/project

# Install requirements 
RUN pip install --no-cache-dir -r requirements.txt

