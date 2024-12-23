# Cervical cancer cell classification API

This API uses the DenseNet121 model, trained on a private cervical cancer cell dataset.
After training, the model is deployed as an API using LitServe and other frameworks.

## Installation and Usage
### 1. Clone the Repository
```bash
git clone https://github.com/.../cervical-cancer-cls-api.git
```

### Navigate to the project directory and set python path
```bash
cd cervical-cancer-cls-api/
export PYTHONPATH=$PYTHONPATH:/cervical-cancer-cls-api
```

### Install requirements
```bash
pip install -r requirements.txt
```

### Run API
```bash
python src/main.py
```

### Test the server
Run the test client:
```bash
python client.py --image ./data/sample.jpg
```
Or use this terminal command:
```bash
curl -X POST "http://127.0.0.1:8000/v1/api/predict" \
-H "Authorization: Bearer ...key..." \
-F "request=@path/to/your/file" \
-k
```

### Check API monitoring metrics
Prometheus metrics available at `v1/api/metrics`.

### 2. Setup Using Docker

To containerize and run the application using Docker, follow these steps:

#### Build the Docker Image
Ensure you are in the root directory of the project where the `Dockerfile` is located. Then, build the Docker image:
```bash
docker build -t cervical-cancer-cls-api .
```

#### Run the Docker Container
Once the image is built, you can run a container using the following command:
```bash
docker run -d -p 8000:8000 --name cervical-cancer-api-container cervical-cancer-cls-api
```
This will expose the API on port 8000.

#### Test the API in the Container
Use the same commands for testing the API as described above, ensuring the host URL points to `http://127.0.0.1:8000`.

#### Stop and Remove the Container
To stop the container, run:
```bash
docker stop cervical-cancer-api-container
```
To remove the container, run:
```bash
docker rm cervical-cancer-api-container
```

### Notes
- Make sure Docker is installed and running on your system before starting.
- If you modify the code, rebuild the image to apply the changes using the `docker build` command.

## Test with unit test