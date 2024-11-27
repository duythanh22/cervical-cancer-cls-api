import unittest
import io
import os
import logging
import requests
from typing import Dict, Any, Optional
from requests.exceptions import RequestException

from config.config import config


class ImageClassificationAPITestBase(unittest.TestCase):
    """Base test class for image classification API tests"""

    @classmethod
    def setUpClass(cls):
        """Set up class-level configurations"""
        cls.api_url = config.API_URL
        cls.api_token = config.API_AUTH_TOKEN
        cls.test_images_dir = "/home/xeon-3/PycharmProjects/deploy-cc-classification/image_test/"

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger(__name__)

    def setUp(self):
        """Set up test-specific configurations"""
        self.session = requests.Session()
        self.session.verify = False

    def load_test_image(self, filename: str) -> io.BytesIO:
        """
        Load test image from file
        """
        filepath = os.path.join(self.test_images_dir, filename)

        if not os.path.exists(filepath):
            self.logger.error(f"Test image not found: {filepath}")
            raise FileNotFoundError(f"Test image not found: {filepath}")

        with open(filepath, "rb") as image_file:
            return io.BytesIO(image_file.read())

    def send_image_request(
            self,
            image: io.BytesIO,
            filename: str,
            headers: Optional[Dict[str, str]] = None
    ) -> requests.Response:
        """
        Send image classification request
        """
        default_headers = {"Authorization": f"Bearer {self.api_token}"}

        if headers:
            default_headers.update(headers)

        files = {"request": (filename, image, f"image/{filename.split('.')[-1]}")}

        try:
            return self.session.post(
                self.api_url,
                files=files,
                headers=default_headers
            )
        except RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise

    def validate_classification_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Validate classification API response
        """
        self.assertEqual(response.status_code, 200, f"Unexpected status code: {response.status_code}")

        json_response = response.json()
        expected_keys = ["predicted_class", "class_probabilities", "confidence_score"]

        for key in expected_keys:
            self.assertIn(key, json_response, f"Missing key: {key}")

        return json_response


class TestImageClassificationAPI(ImageClassificationAPITestBase):
    """Specific test cases for image classification API"""

    def test_successful_image_classification(self):
        """Test successful image classification"""
        try:
            test_image = self.load_test_image("61f01c5fcec542a39807a20c.png")
            response = self.send_image_request(test_image, "61f01c5fcec542a39807a20c.png")
            result = self.validate_classification_response(response)

            # self.logger.info(f"Classification result: {result}")
            self.logger.info("test_successful_image_classification: OK")
        except Exception as e:
            self.logger.error(f"test_successful_image_classification failed: {e}")
            raise

    def test_invalid_token(self):
        """Test API response with invalid authentication token"""
        try:
            test_image = self.load_test_image("6211c2e171c8f45788287393.png")
            headers = {"Authorization": "Bearer invalid_token"}
            response = self.send_image_request(test_image, "6211c2e171c8f45788287393.png", headers)

            self.assertEqual(response.status_code, 401)
            self.assertIn("Invalid or missing token", response.json()["detail"])

            self.logger.info("test_invalid_token: OK")
        except Exception as e:
            self.logger.error(f"test_invalid_token failed: {e}")
            raise

    def test_unsupported_image_format(self):
        """Test API response with unsupported image format"""
        try:
            test_image = self.load_test_image("gif.gif")
            response = self.send_image_request(test_image, "gif.gif")

            self.assertEqual(response.status_code, 400)
            self.assertIn("Unsupported image format", response.json()["detail"])

            self.logger.info("test_unsupported_image_format: OK")
        except Exception as e:
            self.logger.error(f"test_unsupported_image_format failed: {e}")
            raise

    def test_missing_image(self):
        """Test API response when no image is provided"""
        try:
            response = self.session.post(
                self.api_url,
                files={},
                headers={"Authorization": f"Bearer {self.api_token}"}
            )

            self.assertEqual(response.status_code, 422)  # Unprocessable Entity
            self.logger.info("test_missing_image: OK")
        except Exception as e:
            self.logger.error(f"test_missing_image failed: {e}")
            raise

    def test_large_image(self):
        """Test API response with an oversized image"""
        try:
            test_image = self.load_test_image("large.jpg")
            response = self.send_image_request(test_image, "large.jpg")

            self.assertEqual(response.status_code, 413)  # Payload Too Large
            self.logger.info("test_large_image: OK")
        except Exception as e:
            self.logger.error(f"test_large_image failed: {e}")
            raise

    def test_corrupted_image(self):
        """Test API response with various corrupted image scenarios"""
        try:
            # Test 1: Completely invalid file content
            corrupted_image_1 = io.BytesIO(b'not an image')
            response_1 = self.send_image_request(corrupted_image_1, "corrupted_image.png")
            self.assertEqual(response_1.status_code, 400)
            self.assertIn("Invalid or corrupted image file", response_1.json()["detail"])

            self.logger.info("test_corrupted_image: OK")
        except Exception as e:
            self.logger.error(f"test_corrupted_image failed: {e}")
            raise

    def test_multiple_images(self):
        """Test classification of multiple images"""
        try:
            test_images = [
                ("6203368cc4dbb4451b872de6.png", "6203368cc4dbb4451b872de6.png"),
                ("61f6b14a12ad37b9dc961ab3.png", "61f6b14a12ad37b9dc961ab3.png")
            ]

            for filename, image_filename in test_images:
                test_image = self.load_test_image(filename)
                response = self.send_image_request(test_image, image_filename)

                result = self.validate_classification_response(response)
                # self.logger.info(f"Processed {image_filename}: {result}")

            self.logger.info("test_multiple_images: OK")
        except Exception as e:
            self.logger.error(f"test_multiple_images failed: {e}")
            raise


if __name__ == '__main__':
    unittest.main()