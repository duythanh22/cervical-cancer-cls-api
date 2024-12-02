
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# import requests
#
# response = requests.post("http://127.0.0.1:8000/predict", json={"input": 4.0})
# print(f"Status: {response.status_code}\nResponse:\n {response.text}")

import argparse
import requests
from config.config import config

# URL and headers from your configuration
# API_URL = config.API_URL
API_URL = "https://8000-01jds4gkkdm99xkn3an40wbfk6.cloudspaces.litng.ai/v1/api/predict"

def send_request(image_path):
    """Send the image file to the API and return the response."""
    header = {
        "Authorization": f"Bearer {config.API_AUTH_TOKEN}",
    }
    with open(image_path, 'rb') as image_file:
        files = {
            "request": (
                image_path,
                image_file,
                'image/jpeg' if image_path.lower().endswith(('.jpg', '.jpeg')) else 'image/png'
            )
        }
        response = requests.post(API_URL, headers=header, files=files, verify=False)
        response.raise_for_status()
        return response.json()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Send an image to the API for prediction.')
    parser.add_argument("image_path", type=str, help="Path to the image file to classify")
    args = parser.parse_args()

    try:
        result = send_request(args.image_path)
        print("Prediction result:", result)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
