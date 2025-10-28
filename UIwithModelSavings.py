"""
Automated Speech Recognition with Semantic Analysis - Improved Version
This system converts speech to text and analyzes semantic content using neural networks.

Key improvements:
- Model persistence (save/load trained models)
- Simple web UI using Gradio
- Batch audio file processing
- Separate training and inference modes
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import warnings
import pandas as pd
import os
import pickle
from pathlib import Path
import gradio as gr
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration"""
    # Paths
    MODEL_DIR = "saved_models"
    CLASSIFIER_PATH = os.path.join(MODEL_DIR, "semantic_classifier.pth")
    VOCAB_PATH = os.path.join(MODEL_DIR, "vocabulary.pkl")

    # Model parameters
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    MAX_LEN = 100

    # Training parameters
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2

    # Whisper model
    WHISPER_MODEL_SIZE = "base"  # 'tiny', 'base', 'small', 'medium', 'large'

    @staticmethod
    def ensure_model_dir():
        """Create model directory if it doesn't exist"""
        Path(Config.MODEL_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================================
# SPEECH-TO-TEXT
# ============================================================================

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


# ============================================================================
# SEMANTIC CLASSIFIER
# ============================================================================

class SemanticClassifier(nn.Module):
    """Neural Network for semantic text analysis"""

    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=256, num_classes=5):
        super(SemanticClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)

        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)

        x = self.fc1(attended)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class TextPreprocessor:
    """Converts text to numerical format for neural network"""

    def __init__(self, max_words=10000, max_len=100):
        self.max_words = max_words
        self.max_len = max_len
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}

    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        word_freq = {}
        for text in texts:
            for word in text.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, _) in enumerate(sorted_words[:self.max_words-2], start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        words = text.lower().split()
        sequence = [self.word_to_idx.get(word, 1) for word in words[:self.max_len]]

        if len(sequence) < self.max_len:
            sequence += [0] * (self.max_len - len(sequence))

        return torch.tensor(sequence, dtype=torch.long)

    def save(self, path):
        """Save preprocessor state"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'max_words': self.max_words,
                'max_len': self.max_len
            }, f)
        print(f"Vocabulary saved to {path}")

    def load(self, path):
        """Load preprocessor state"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word_to_idx = data['word_to_idx']
            self.idx_to_word = data['idx_to_word']
            self.max_words = data['max_words']
            self.max_len = data['max_len']
        print(f"Vocabulary loaded from {path}")


# ============================================================================
# INTEGRATED SYSTEM
# ============================================================================

class SpeechSemanticAnalyzer:
    """Complete system: Speech -> Text -> Semantic Analysis"""

    def __init__(self, categories=None):
        self.categories = categories
        self.stt = None
        self.preprocessor = TextPreprocessor(
            max_words=Config.VOCAB_SIZE,
            max_len=Config.MAX_LEN
        )
        self.classifier = None

    def load_speech_model(self, model_size=None):
        """Load speech-to-text model"""
        if model_size is None:
            model_size = Config.WHISPER_MODEL_SIZE
        self.stt = SpeechToText(model_size)

    def train_classifier(self, texts, labels, epochs=None, batch_size=None):
        """Train the semantic classifier"""
        if epochs is None:
            epochs = Config.EPOCHS
        if batch_size is None:
            batch_size = Config.BATCH_SIZE

        print("Training semantic classifier...")

        # Build vocabulary
        self.preprocessor.build_vocab(texts)

        # Initialize classifier
        if self.classifier is None:
            self.classifier = SemanticClassifier(
                vocab_size=Config.VOCAB_SIZE,
                embedding_dim=Config.EMBEDDING_DIM,
                hidden_dim=Config.HIDDEN_DIM,
                num_classes=len(self.categories)
            )

        # Prepare data
        sequences = torch.stack([self.preprocessor.text_to_sequence(text) for text in texts])
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Split into train and validation
        indices = torch.randperm(len(sequences))
        split_idx = int(len(sequences) * (1 - Config.VALIDATION_SPLIT))

        train_sequences = sequences[indices[:split_idx]]
        train_labels = labels_tensor[indices[:split_idx]]
        val_sequences = sequences[indices[split_idx:]]
        val_labels = labels_tensor[indices[split_idx:]]

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=Config.LEARNING_RATE)

        # Training loop
        self.classifier.train()
        best_val_acc = 0.0

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for i in range(0, len(train_sequences), batch_size):
                batch_seq = train_sequences[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.classifier(batch_seq)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (outputs.argmax(1) == batch_labels).sum().item()
                total += batch_labels.size(0)

            # Validation
            self.classifier.eval()
            with torch.no_grad():
                val_outputs = self.classifier(val_sequences)
                val_loss = criterion(val_outputs, val_labels)
                val_accuracy = (val_outputs.argmax(1) == val_labels).float().mean().item()
            self.classifier.train()

            train_accuracy = correct / total

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy

            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train - Loss: {total_loss/len(train_sequences)*batch_size:.4f}, Accuracy: {train_accuracy:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        self.classifier.eval()
        print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")

    def save_model(self):
        """Save trained model and vocabulary"""
        Config.ensure_model_dir()

        # Save classifier
        torch.save({
            'model_state_dict': self.classifier.state_dict(),
            'categories': self.categories,
            'vocab_size': Config.VOCAB_SIZE,
            'embedding_dim': Config.EMBEDDING_DIM,
            'hidden_dim': Config.HIDDEN_DIM,
        }, Config.CLASSIFIER_PATH)
        print(f"Model saved to {Config.CLASSIFIER_PATH}")

        # Save vocabulary
        self.preprocessor.save(Config.VOCAB_PATH)

    def load_model(self):
        """Load trained model and vocabulary"""
        if not os.path.exists(Config.CLASSIFIER_PATH):
            raise FileNotFoundError(f"No saved model found at {Config.CLASSIFIER_PATH}")

        # Load classifier
        checkpoint = torch.load(Config.CLASSIFIER_PATH)
        self.categories = checkpoint['categories']

        # Convert numeric categories to readable labels if needed
        self.categories = self._format_categories(self.categories)

        self.classifier = SemanticClassifier(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_classes=len(self.categories)
        )
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()
        print(f"Model loaded from {Config.CLASSIFIER_PATH}")

        # Load vocabulary
        self.preprocessor.load(Config.VOCAB_PATH)

    def _format_categories(self, categories):
        """Convert categories to readable format"""
        # Mapping for common sentiment labels
        label_map = {
            0: "Negative intention",
            1: "Neutral intention",
            2: "Positive intention",
            '0': "Negative intention",
            '1': "Neutral intention",
            '2': "Positive intention",
            'positive': "Positive intention",
            'negative': "Negative intention",
            'neutral': "Neutral intention",
            'Positive': "Positive intention",
            'Negative': "Negative intention",
            'Neutral': "Neutral intention"
        }

        formatted = []
        for cat in categories:
            if cat in label_map:
                formatted.append(label_map[cat])
            else:
                # If not in map, convert to string and capitalize
                formatted.append(str(cat).capitalize() + " intention")

        return formatted

    def analyze_text(self, text):
        """Analyze text semantics only"""
        if self.classifier is None:
            raise ValueError("Classifier not loaded. Call load_model() or train_classifier() first.")

        sequence = self.preprocessor.text_to_sequence(text).unsqueeze(0)

        with torch.no_grad():
            output = self.classifier(sequence)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = output.argmax(1).item()

        return {
            "category": self.categories[predicted_class],
            "probabilities": {cat: prob.item() for cat, prob in zip(self.categories, probabilities)},
            "confidence": probabilities[predicted_class].item()
        }

    def analyze_audio(self, audio_path, language="english"):
        """Full pipeline: audio -> text -> semantic category"""
        if self.stt is None:
            self.load_speech_model()

        if self.classifier is None:
            raise ValueError("Classifier not loaded. Call load_model() or train_classifier() first.")

        # Step 1: Speech to text
        transcription = self.stt.transcribe(audio_path, language)

        # Step 2: Semantic analysis
        analysis = self.analyze_text(transcription)
        analysis["transcription"] = transcription

        return analysis


# ============================================================================
# GRADIO UI
# ============================================================================

def create_ui(analyzer):
    """Create Gradio interface"""

    def process_audio(audio_file, language):
        """Process uploaded audio file"""
        try:
            result = analyzer.analyze_audio(audio_file, language)

            # Format probabilities
            prob_text = "\n".join([f"**{cat}**: {prob:.3f}" for cat, prob in result['probabilities'].items()])

            return (
                result['transcription'],
                result['category'],
                f"{result['confidence']:.3f}",
                prob_text
            )
        except Exception as e:
            return str(e), "Error", "0.000", ""

    def process_text(text):
        """Process text input directly"""
        try:
            result = analyzer.analyze_text(text)
            prob_text = "\n".join([f"**{cat}**: {prob:.3f}" for cat, prob in result['probabilities'].items()])

            return (
                result['category'],
                f"{result['confidence']:.3f}",
                prob_text
            )
        except Exception as e:
            return "Error", "0.000", str(e)

    # Create interface
    with gr.Blocks(title="Speech Semantic Analyzer") as demo:
        gr.Markdown("# 🎤 Automated Speech Recognition with Semantic Analysis")
        gr.Markdown("Upload an audio file or enter text to analyze its semantic content")

        with gr.Tab("Audio Analysis"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(type="filepath", label="Upload Audio File")
                    language_input = gr.Dropdown(
                        choices=["english", "russian", "spanish", "french", "german"],
                        value="english",
                        label="Language"
                    )
                    audio_btn = gr.Button("Analyze Audio", variant="primary")

                with gr.Column():
                    transcription_output = gr.Textbox(label="Transcription", lines=3)
                    category_output = gr.Textbox(label="Predicted Category")
                    confidence_output = gr.Textbox(label="Confidence")
                    probs_output = gr.Markdown(label="All Probabilities")

            audio_btn.click(
                fn=process_audio,
                inputs=[audio_input, language_input],
                outputs=[transcription_output, category_output, confidence_output, probs_output]
            )

        with gr.Tab("Text Analysis"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(label="Enter Text", lines=4, placeholder="Type or paste text here...")
                    text_btn = gr.Button("Analyze Text", variant="primary")

                with gr.Column():
                    text_category_output = gr.Textbox(label="Predicted Category")
                    text_confidence_output = gr.Textbox(label="Confidence")
                    text_probs_output = gr.Markdown(label="All Probabilities")

            text_btn.click(
                fn=process_text,
                inputs=[text_input],
                outputs=[text_category_output, text_confidence_output, text_probs_output]
            )

        gr.Markdown(f"### Model Info\n**Categories**: {', '.join(map(str, analyzer.categories))}")

    return demo


# ============================================================================
# DATASET LOADING (Same as before)
# ============================================================================

def load_kaggle_sentiment_dataset(dataset_path):
    """Load sentiment dataset from Kaggle"""
    print(f"Loading dataset from: {dataset_path}")

    csv_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")

    all_texts = []
    all_labels = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        text_col = None
        label_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'text' in col_lower or 'review' in col_lower or 'sentence' in col_lower:
                text_col = col
            if 'sentiment' in col_lower or 'label' in col_lower or 'rating' in col_lower:
                label_col = col

        if text_col is None or label_col is None:
            text_col = df.columns[0]
            label_col = df.columns[1]

        texts = df[text_col].astype(str).tolist()
        labels_raw = df[label_col].tolist()

        all_texts.extend(texts)
        all_labels.extend(labels_raw)

    unique_labels = sorted(list(set(all_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_to_idx[label] for label in all_labels]

    print(f"Total samples: {len(all_texts)}")
    print(f"Categories: {unique_labels}")

    return all_texts, numeric_labels, unique_labels


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def train_mode():
    """Training mode - train and save model"""
    print("\n" + "="*60)
    print("TRAINING MODE")
    print("="*60 + "\n")

    # Load dataset
    try:
        import kagglehub
        print("Downloading Kaggle dataset...")
        dataset_path = kagglehub.dataset_download("akgeni/generic-sentiment-multidomain-sentiment-dataset")
        texts, labels, categories = load_kaggle_sentiment_dataset(dataset_path)

        # Limit for faster training
        MAX_SAMPLES = 5000
        if len(texts) > MAX_SAMPLES:
            print(f"Limiting to {MAX_SAMPLES} samples...")
            texts = texts[:MAX_SAMPLES]
            labels = labels[:MAX_SAMPLES]
    except:
        print("Using example data...")
        categories = ['positive', 'negative', 'neutral']
        texts = [
            "this is amazing I love it so much",
            "terrible experience very disappointed",
            "it's okay nothing special",
            "absolutely wonderful highly recommend",
            "worst product ever waste of money",
        ] * 20
        labels = [0, 1, 2, 0, 1] * 20

    # Train model
    analyzer = SpeechSemanticAnalyzer(categories)
    analyzer.train_classifier(texts, labels)

    # Save model
    analyzer.save_model()

    print("\n✅ Training complete! Model saved to", Config.MODEL_DIR)
    print("Run in inference mode to use the trained model.")


def inference_mode():
    """Inference mode - load model and start UI"""
    print("\n" + "="*60)
    print("INFERENCE MODE")
    print("="*60 + "\n")

    # Load model
    analyzer = SpeechSemanticAnalyzer()
    try:
        analyzer.load_model()
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ No trained model found!")
        print("Please run in training mode first (MODE = 'train')")
        return

    # Create and launch UI
    print("\nLaunching Gradio interface...")
    demo = create_ui(analyzer)
    demo.launch(share=False)


if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION: Set mode here
    # ========================================================================

    MODE = "inference"  # Change to "train" to train a new model

    if MODE == "train":
        train_mode()
    elif MODE == "inference":
        inference_mode()
    else:
        print("Invalid MODE. Set MODE to 'train' or 'inference'")