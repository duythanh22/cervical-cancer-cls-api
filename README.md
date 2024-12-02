# Cervical cancer cell classification API

This API uses the DenseNet121 model, trained on a private cervical cancer cell dataset. 
After training, the model is deployed as an API using LitServe and other frameworks.

## Installation and Usage
### Clone the Repository
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

### Run api
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

### Check api monitoring metrics
Prometheus metrics available at ```v1/api/metrics```:

### Test with unit test

