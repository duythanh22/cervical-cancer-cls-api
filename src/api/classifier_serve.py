import torch
import PIL
import os
from loguru import logger
import litserve as ls
from torchvision import transforms
from fastapi import Depends, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from litserve.utils import PickleableHTTPException
from src.core.config import config
from typing import List
import psutil
import torch.cuda as cuda
import time
import PIL.Image

class CervicalCellClassifierAPI(ls.LitAPI):
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
            image = PIL.Image.open(request.file)

            file_ext = os.path.splitext(request.filename)[1].lower()
            if file_ext not in ['.png', '.jpg', '.jpeg']:
                raise PickleableHTTPException(status_code=400,
                                              detail=f"Unsupported image format: {file_ext}")

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
            logger.error(f"Error in decode_request: {e}")
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
                self.log("system_memory_total", psutil.virtual_memory().total)
                self.log("system_memory_available", psutil.virtual_memory().available)
                self.log("system_memory_used", psutil.virtual_memory().used)

                if allocated is not None:
                    self.log("model_memory_allocated", allocated * (1024 ** 3))
                    self.log("model_memory_peak", peak * (1024 ** 3))

            except Exception as mem_err:
                logger.warning(f"Failed to retrieve system metrics: {mem_err}")

            self.log("inference_time", time_taken)

            predicted_class = config.CLASS_NAMES[predicted.item()]
            confidence = probabilities[0][predicted.item()].item()
            self.log("model_prediction", (predicted_class, confidence))

            logger.info(f"Prediction completed in {time_taken:.4f} seconds")
            logger.debug(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")

            return predicted, probabilities

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise PickleableHTTPException(status_code=500, detail="Prediction failed")

    def unbatch(self, output):
        predicted, probabilities = output
        return list(zip(predicted, probabilities))

    def encode_response(self, output, **kwargs):
        predicted_label, probabilities = output

        def process_prediction(label, probs):
            label_idx = label.item()
            if 0 <= label_idx < len(config.CLASS_NAMES):
                predicted_class = config.CLASS_NAMES[label_idx]
                class_probabilities = {config.CLASS_NAMES[i]: prob.item() for i, prob in enumerate(probs)}
                confidence_score = probs[label_idx].item()

                return {
                    "predicted_class": predicted_class,
                    "class_probabilities": class_probabilities,
                    "confidence_score": confidence_score
                }

            logger.error(f"Invalid prediction index: {label_idx}")
            raise PickleableHTTPException(status_code=500, detail="Invalid model prediction")

        if predicted_label.dim() > 0:
            return [process_prediction(label, probs) for label, probs in zip(predicted_label, probabilities)]
        else:
            return process_prediction(predicted_label, probabilities)

    @staticmethod
    def authorize(credentials: HTTPAuthorizationCredentials = Depends(security)):
        token = credentials.credentials
        if token != config.API_AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid or missing token")