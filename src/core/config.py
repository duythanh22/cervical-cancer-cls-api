from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_AUTH_TOKEN: str = "cc-classify"
    MODEL_PATH: str = 'src/model/best_model_checkpoint.pt'
    SERVER_PORT: int = 8000
    MAX_IMAGE_SIZE: int = 5 * 1024 * 1024  # 10MB
    IMAGE_INPUT_SIZE: tuple = (224, 224)
    IMAGE_MEAN: tuple = (0.485, 0.456, 0.406)
    IMAGE_STD: tuple = (0.229, 0.224, 0.225)
    CLASS_NAMES: list = ["ASC_H", "ASC_US", "HSIL", "LSIL", "SCC"]  # replace with actual class names

    class Config:
        env_file = ".env"

config = Settings()