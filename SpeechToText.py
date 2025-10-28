from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch

class SpeechToText:
    """Handles speech-to-text conversion using Whisper model"""

    def __init__(self, model_size="base"):
        print(f"Loading Whisper {model_size} model...")
        self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")
        self.model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}")
        self.model.eval()

    def transcribe(self, audio_path, language="english"):
        """Transcribe audio file to text"""
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Process audio
        input_features = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language=language,
                task="transcribe"
            )

        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return transcription

