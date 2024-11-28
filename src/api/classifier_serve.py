import torch
import PIL
import os
from loguru import logger
import litserve as ls
from torchvision import transforms
from fastapi import Depends, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from litserve.utils import PickleableHTTPException
from config.config import config
from typing import List
import psutil
import torch.cuda as cuda
import time
import PIL.Image

class CervicalCellClassifierAPI(ls.LitAPI):
    """
    CervicalCellClassifierAPI is a LitServe-based service for classifying cervical cell images.

    This class extends the LitAPI from the litserve library and provides methods for setting up
    the model, decoding image requests, batching inputs, making predictions, unbatching outputs,
    and encoding responses. It also includes an authorization mechanism using HTTPBearer security.

    Attributes:
        security (HTTPBearer): Security scheme for API authentication.

    Methods:
        setup(devices): Loads the model onto the specified device(s) and prepares it for inference.
        decode_request(request, **kwargs): Validates and processes an uploaded image file into a tensor.
        batch(inputs): Combines a list of tensors into a single batch tensor.
        predict(x, **kwargs): Performs inference on the input tensor and logs system metrics.
        unbatch(output): Separates the batched output into individual predictions.
        encode_response(output, **kwargs): Formats the prediction results into a JSON-compatible response.
        authorize(credentials): Validates the API token for request authorization.
    """
    security = HTTPBearer()

    def setup(self, devices):
        self.device = devices[0] if isinstance(devices, list) else devices
        try:
            self.model = torch.load(config.MODEL_PATH, map_location=self.device)
            logger.info("Setup completed successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise PickleableHTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        self.model.eval()

    def decode_request(self, request: UploadFile, **kwargs) -> torch.Tensor:
        try:
            # Validate file extension first
            file_ext = os.path.splitext(request.filename)[1].lower()
            if file_ext not in ['.png', '.jpg', '.jpeg']:
                raise PickleableHTTPException(status_code=400,
                                              detail=f"Unsupported image format: {file_ext}")

            # Attempt to open and validate image
            try:
                # Seek to the beginning of the file to ensure we're at the start
                request.file.seek(0)

                # Use verify() to check image integrity before opening
                image_verify = PIL.Image.open(request.file)
                image_verify.verify()

                # Seek back to the beginning for actual image processing
                request.file.seek(0)
                image = PIL.Image.open(request.file)
            except (PIL.UnidentifiedImageError, SyntaxError) as img_err:
                logger.error(f"Image verification failed: {img_err}")
                raise PickleableHTTPException(status_code=400,
                                              detail="Invalid or corrupted image file")

            # Process the image
            image = image.convert("RGB")
            image = image.resize(config.IMAGE_INPUT_SIZE, PIL.Image.NEAREST)

            tensor_image = transforms.ToTensor()(image)
            tensor_image = transforms.Normalize(
                mean=config.IMAGE_MEAN,
                std=config.IMAGE_STD
            )(tensor_image)

            return tensor_image.unsqueeze(0).to(self.device, non_blocking=True)

        except PickleableHTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in decode_request: {e}")
            raise PickleableHTTPException(status_code=500, detail="Image processing failed")

    def batch(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(inputs, dim=0).to(self.device)

    def predict(self, x: torch.Tensor, **kwargs):
        try:
            logger.info("Starting prediction...")
            start_time = time.time()

            with torch.inference_mode():
                output = self.model(x)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)

            end_time = time.time()
            time_taken = end_time - start_time

            try:
                if cuda.is_available():
                    allocated = cuda.memory_allocated(self.device) / (1024 ** 3)
                    peak = cuda.max_memory_allocated(self.device) / (1024 ** 3)
                else:
                    allocated = None
                    peak = None

                process = psutil.Process(os.getpid())
                cpu_usage = {
                    "system": psutil.cpu_percent(interval=None),
                    "process": process.cpu_percent(interval=None)
                }

                self.log("cpu_usage", cpu_usage)
                self.log("system_memory_total", psutil.virtual_memory().total / (1024 ** 3))
                self.log("system_memory_available", psutil.virtual_memory().available / (1024 ** 3))
                self.log("system_memory_used", psutil.virtual_memory().used / (1024 ** 3))

                if allocated is not None:
                    self.log("model_memory_allocated", allocated)
                    self.log("model_memory_peak", peak)

            except Exception as mem_err:
                logger.warning(f"Failed to retrieve system metrics: {mem_err}")

            self.log("inference_time", time_taken)

            # Handle both batch and single predictions
            for i in range(len(predicted)):
                predicted_class = config.CLASS_NAMES[predicted[i].item()]
                confidence = probabilities[i][predicted[i].item()].item()
                self.log("model_prediction", (predicted_class, confidence))

            logger.info(f"Prediction completed in {time_taken:.4f} seconds")
            return predicted, probabilities

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise PickleableHTTPException(status_code=500, detail="Prediction failed")

    def unbatch(self, output):
        predicted, probabilities = output
        return [(pred, prob) for pred, prob in zip(predicted, probabilities)]

    def encode_response(self, output, **kwargs):
        predicted_label, probabilities = output

        def process_prediction(label, probs):
            label_idx = label.item()
            if 0 <= label_idx < len(config.CLASS_NAMES):
                predicted_class = config.CLASS_NAMES[label_idx]
                class_probabilities = {config.CLASS_NAMES[i]: prob.item()
                                       for i, prob in enumerate(probs)}
                confidence_score = probs[label_idx].item()

                return {
                    "predicted_class": predicted_class,
                    "class_probabilities": class_probabilities,
                    "confidence_score": confidence_score
                }

            logger.error(f"Invalid prediction index: {label_idx}")
            raise PickleableHTTPException(status_code=500, detail="Invalid model prediction")

        # Handle both single and batch predictions
        if isinstance(predicted_label, torch.Tensor) and predicted_label.dim() > 0:
            return [process_prediction(label, probs)
                    for label, probs in zip(predicted_label, probabilities)]
        else:
            return process_prediction(predicted_label, probabilities)

    @staticmethod
    def authorize(credentials: HTTPAuthorizationCredentials = Depends(security)):
        token = credentials.credentials
        if token != config.API_AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid or missing token")