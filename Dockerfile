# Use the latest Ubuntu image
FROM ubuntu:latest
LABEL authors="xeon-3"

# Update and install Python3, pip3, python3-venv and necessary packages
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

# Copy files and folders into container
COPY /src /cervical-cancer-cls-api
COPY /test /cervical-cancer-cls-api
COPY /config /cervical-cancer-cls-api
COPY client.py /cervical-cancer-cls-api
COPY requirements.txt /cervical-cancer-cls-api
COPY /app /cervical-cancer-cls-api

# Set working directory
WORKDIR /cervical-cancer-cls-api

# Create a virtual environment and activate it
RUN python3 -m venv venv

# Install dependencies
RUN venv/bin/pip install --no-cache-dir -r requirements.txt

# Creatte log folder
RUN mkdir -p /cervical-cancer-cls-api/logs

# Set working directory for the server
WORKDIR /cervical-cancer-cls-api/src

# Run server
CMD ["python", "main.py"]
