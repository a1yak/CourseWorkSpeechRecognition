import os
from pathlib import Path

class Config:
    """Central configuration"""
    # Paths
    MODEL_DIR = "saved_models"
    CLASSIFIER_PATH = os.path.join(MODEL_DIR, "semantic_classifier.pth")
    VOCAB_PATH = os.path.join(MODEL_DIR, "vocabulary.pkl")

    # Model parameters
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128
    MAX_LEN = 100

    # Training parameters
    EPOCHS = 8
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2

    # Whisper model
    WHISPER_MODEL_SIZE = "base"  # 'tiny', 'base', 'small', 'medium', 'large'

    @staticmethod
    def ensure_model_dir():
        """Create model directory if it doesn't exist"""
        Path(Config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
