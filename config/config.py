from pydantic_settings import BaseSettings
import os
import secrets

def generate_secret_key(length=32):
    return secrets.token_hex(length)

class Settings(BaseSettings):
    API_AUTH_TOKEN: str = "cc-classify"
    MODEL_PATH: str = 'src/model/best_model_checkpoint.pt'
    API_URL: str = "http://localhost:8000/api/v1/predict"
    SERVER_PORT: int = 8000
    MAX_IMAGE_SIZE: int = 5 * 1024 * 1024  # 10MB
    IMAGE_INPUT_SIZE: tuple = (224, 224)
    IMAGE_MEAN: tuple = (0.485, 0.456, 0.406)
    IMAGE_STD: tuple = (0.229, 0.224, 0.225)
    CLASS_NAMES: list = ["ASC_H", "ASC_US", "HSIL", "LSIL", "SCC"]  # replace with actual class names

    # Flask
    SECRET_KEY: str = os.environ.get('FLASK_SECRET_KEY') or generate_secret_key()
    FLASK_HOST: str = '0.0.0.0'
    FLASK_PORT: int = 5000
    FLASK_DEBUG: bool = False
    class Config:
        env_file = ".env"

config = Settings()