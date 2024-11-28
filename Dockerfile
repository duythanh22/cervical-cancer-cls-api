# Use the latest Ubuntu image
FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04
LABEL authors="xeon-3"

# Set the desired Python version as an argument
ARG PYTHON_VERSION=3.10

# Update and install necessary packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set PYTHONPATH environment variable
ENV PYTHONPATH=$PYTHONPATH:/cervical-cancer-cls-api

# Set working directory
WORKDIR /cervical-cancer-cls-api

# Copy application files into the container
COPY /src ./src
COPY /test ./test
COPY /config ./config
COPY client.py .
COPY requirements.txt .
COPY /app ./app

# Install dependencies globally in the container
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Create a folder for logs
RUN mkdir -p logs

# Set working directory for the server
WORKDIR /cervical-cancer-cls-api/

# Specify the default command to run the server
CMD ["python3", "src/main.py"]