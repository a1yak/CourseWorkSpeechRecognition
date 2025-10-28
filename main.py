"""
Automated Speech Recognition with Semantic Analysis
This system converts speech to text and analyzes semantic content using neural networks.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import warnings
import pandas as pd
import os
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: SPEECH-TO-TEXT (Using Pre-trained Whisper)
# ============================================================================

class SpeechToText:
    """Handles speech-to-text conversion using Whisper model"""

    def __init__(self, model_size="base"):
        """
        Initialize Whisper model
        Args:
            model_size: 'tiny', 'base', 'small', 'medium', 'large'
        """
        print(f"Loading Whisper {model_size} model...")
        self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")
        self.model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}")
        self.model.eval()

    def transcribe(self, audio_path, language="russian"):
        """
        Transcribe audio file to text
        Args:
            audio_path: path to audio file (.wav, .mp3, etc)
            language: 'russian', 'english', etc.
        Returns:
            transcribed text
        """
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
# PART 2: SEMANTIC ANALYSIS (Your Neural Network)
# ============================================================================

class SemanticClassifier(nn.Module):
    """
    Neural Network for semantic text analysis
    This is YOUR contribution - a custom NN for text classification
    """

    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=256, num_classes=5):
        """
        Args:
            vocab_size: size of vocabulary
            embedding_dim: dimension of word embeddings
            hidden_dim: hidden layer size
            num_classes: number of semantic categories
        """
        super(SemanticClassifier, self).__init__()

        # Embedding layer (converts words to vectors)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass
        Args:
            x: tensor of word indices [batch_size, seq_length]
        Returns:
            class probabilities [batch_size, num_classes]
        """
        # Embedding
        embedded = self.embedding(x)  # [batch, seq, embed_dim]

        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(embedded)  # [batch, seq, hidden*2]

        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)  # [batch, hidden*2]

        # Classification layers
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

        # Keep most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, _) in enumerate(sorted_words[:self.max_words-2], start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        words = text.lower().split()
        sequence = [self.word_to_idx.get(word, 1) for word in words[:self.max_len]]

        # Pad sequence
        if len(sequence) < self.max_len:
            sequence += [0] * (self.max_len - len(sequence))

        return torch.tensor(sequence, dtype=torch.long)


# ============================================================================
# PART 3: INTEGRATED SYSTEM
# ============================================================================

class SpeechSemanticAnalyzer:
    """Complete system: Speech -> Text -> Semantic Analysis"""

    def __init__(self, categories):
        """
        Args:
            categories: list of semantic categories (e.g., ['positive', 'negative', 'neutral', 'question', 'command'])
        """
        self.categories = categories
        self.stt = None  # Lazy loading
        self.preprocessor = TextPreprocessor()
        self.classifier = SemanticClassifier(num_classes=len(categories))

    def load_speech_model(self, model_size="base"):
        """Load speech-to-text model"""
        self.stt = SpeechToText(model_size)

    def train_classifier(self, texts, labels, epochs=10, batch_size=32, validation_split=0.2):
        """
        Train the semantic classifier
        Args:
            texts: list of text samples
            labels: list of category indices
            epochs: number of training epochs
            batch_size: batch size for training
            validation_split: fraction of data to use for validation
        """
        print("Training semantic classifier...")

        # Build vocabulary
        self.preprocessor.build_vocab(texts)

        # Prepare data
        sequences = torch.stack([self.preprocessor.text_to_sequence(text) for text in texts])
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Split into train and validation
        indices = torch.randperm(len(sequences))
        split_idx = int(len(sequences) * (1 - validation_split))

        train_sequences = sequences[indices[:split_idx]]
        train_labels = labels_tensor[indices[:split_idx]]
        val_sequences = sequences[indices[split_idx:]]
        val_labels = labels_tensor[indices[split_idx:]]

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)

        # Training loop
        self.classifier.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            # Mini-batch training
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
                val_accuracy = (val_outputs.argmax(1) == val_labels).float().mean()
            self.classifier.train()

            train_accuracy = correct / total
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train - Loss: {total_loss/len(train_sequences)*batch_size:.4f}, Accuracy: {train_accuracy:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        self.classifier.eval()
        print("Training complete!")

    def analyze_audio(self, audio_path, language="russian"):
        """
        Full pipeline: audio -> text -> semantic category
        Args:
            audio_path: path to audio file
            language: language of speech
        Returns:
            dict with transcription and semantic analysis
        """
        # Step 1: Speech to text
        if self.stt is None:
            self.load_speech_model()

        transcription = self.stt.transcribe(audio_path, language)

        # Step 2: Semantic analysis
        sequence = self.preprocessor.text_to_sequence(transcription).unsqueeze(0)

        with torch.no_grad():
            output = self.classifier(sequence)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = output.argmax(1).item()

        return {
            "transcription": transcription,
            "category": self.categories[predicted_class],
            "probabilities": {cat: prob.item() for cat, prob in zip(self.categories, probabilities)}
        }


# ============================================================================
# DATASET LOADING UTILITIES
# ============================================================================

def load_kaggle_sentiment_dataset(dataset_path):
    """
    Load the Generic Sentiment Multi-Domain dataset from Kaggle
    Args:
        dataset_path: path where kagglehub downloaded the dataset
    Returns:
        texts (list), labels (list), categories (list)
    """
    print(f"Loading dataset from: {dataset_path}")

    # Find CSV files in the downloaded path
    csv_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")

    print(f"Found {len(csv_files)} CSV file(s)")

    # Load the dataset
    # This dataset typically has columns: 'text' and 'sentiment' (or similar)
    all_texts = []
    all_labels = []

    for csv_file in csv_files:
        print(f"Loading: {csv_file}")
        df = pd.read_csv(csv_file)

        # Display first few rows to understand structure
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        print(df.head(2))

        # Adapt column names (common variations)
        text_col = None
        label_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'text' in col_lower or 'review' in col_lower or 'sentence' in col_lower:
                text_col = col
            if 'sentiment' in col_lower or 'label' in col_lower or 'rating' in col_lower:
                label_col = col

        if text_col is None or label_col is None:
            print(f"Warning: Could not auto-detect columns. Using first two columns.")
            text_col = df.columns[0]
            label_col = df.columns[1]

        print(f"Using text column: '{text_col}', label column: '{label_col}'")

        # Extract texts and labels
        texts = df[text_col].astype(str).tolist()
        labels_raw = df[label_col].tolist()

        all_texts.extend(texts)
        all_labels.extend(labels_raw)

    # Convert labels to numeric indices
    unique_labels = sorted(list(set(all_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_to_idx[label] for label in all_labels]

    print(f"\nDataset loaded successfully!")
    print(f"Total samples: {len(all_texts)}")
    print(f"Categories: {unique_labels}")
    print(f"Label distribution: {pd.Series(all_labels).value_counts().to_dict()}")

    return all_texts, numeric_labels, unique_labels


def load_custom_csv_dataset(csv_path, text_column='text', label_column='label'):
    """
    Load a custom CSV dataset with text and labels
    Args:
        csv_path: path to CSV file
        text_column: name of column containing text
        label_column: name of column containing labels
    Returns:
        texts (list), labels (list), categories (list)
    """
    print(f"Loading dataset from: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")

    texts = df[text_column].astype(str).tolist()
    labels_raw = df[label_column].tolist()

    # Convert labels to numeric
    unique_labels = sorted(list(set(labels_raw)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_to_idx[label] for label in labels_raw]

    print(f"Categories: {unique_labels}")
    print(f"Label distribution: {pd.Series(labels_raw).value_counts().to_dict()}")

    return texts, numeric_labels, unique_labels


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":

    # ========================================================================
    # OPTION 1: Load from Kaggle dataset
    # ========================================================================
    USE_KAGGLE_DATASET = True  # Set to True to use Kaggle dataset

    if USE_KAGGLE_DATASET:
        try:
            import kagglehub

            # Download dataset
            print("Downloading Kaggle dataset...")
            dataset_path = kagglehub.dataset_download("akgeni/generic-sentiment-multidomain-sentiment-dataset")

            # Load the dataset
            texts, labels, categories = load_kaggle_sentiment_dataset(dataset_path)

            # Optional: Limit dataset size for faster training
            MAX_SAMPLES = 5000
            if len(texts) > MAX_SAMPLES:
                print(f"Limiting to {MAX_SAMPLES} samples for faster training...")
                texts = texts[:MAX_SAMPLES]
                labels = labels[:MAX_SAMPLES]

        except ImportError:
            print("kagglehub not installed. Install with: pip install kagglehub")
            print("Falling back to example data...")
            USE_KAGGLE_DATASET = False
        except Exception as e:
            print(f"Error loading Kaggle dataset: {e}")
            print("Falling back to example data...")
            USE_KAGGLE_DATASET = False

    # ========================================================================
    # OPTION 2: Use small example data (fallback)
    # ========================================================================
    if not USE_KAGGLE_DATASET:
        categories = ['positive', 'negative', 'neutral']

        texts = [
            "this is amazing I love it so much",
            "terrible experience very disappointed",
            "it's okay nothing special",
            "absolutely wonderful highly recommend",
            "worst product ever waste of money",
            "average quality meets expectations",
            "fantastic great value for money",
            "horrible never buying again",
            "decent product does the job",
            "excellent quality very satisfied",
        ] * 10  # Repeat for more samples

        labels = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0] * 10

    # ========================================================================
    # TRAIN THE MODEL
    # ========================================================================

    print(f"\n{'='*60}")
    print(f"Training semantic classifier with {len(texts)} samples")
    print(f"Categories: {categories}")
    print(f"{'='*60}\n")

    # Initialize system
    analyzer = SpeechSemanticAnalyzer(categories)

    # Train the classifier
    analyzer.train_classifier(
        texts,
        labels,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )

    # ========================================================================
    # TEST THE MODEL
    # ========================================================================

    print("\n" + "="*60)
    print("Testing on sample texts")
    print("="*60)

    test_texts = [
        "this product is absolutely fantastic",
        "terrible quality very disappointed",
        "it works fine nothing extraordinary"
    ]

    for test_text in test_texts:
        test_sequence = analyzer.preprocessor.text_to_sequence(test_text).unsqueeze(0)
        with torch.no_grad():
            output = analyzer.classifier(test_sequence)
            probs = torch.softmax(output, dim=1)[0]
            pred = output.argmax(1).item()

        print(f"\nText: '{test_text}'")
        print(f"Predicted: {categories[pred]}")
        print(f"Probabilities: {dict(zip(categories, [f'{p:.3f}' for p in probs]))}")

    # ========================================================================
    # AUDIO ANALYSIS (uncomment when you have audio file)
    # ========================================================================

    print("\n" + "="*60)
    print("To analyze audio files, use:")
    print("="*60)
    print("result = analyzer.analyze_audio('path/to/audio.wav', language='english')")
    print("print(result)")


    result = analyzer.analyze_audio("C:/Users/anton/Downloads/28e563cc-d795-4f59-a53f-75bf0f3216c1.wav", language="english")
    print(f"\nTranscription: {result['transcription']}")
    print(f"Category: {result['category']}")
    print(f"Probabilities: {result['probabilities']}")