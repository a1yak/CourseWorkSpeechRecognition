from Config import Config
from SemanticClassifier import SemanticClassifier
from SpeechToText import SpeechToText
from TextPreprocesssor import TextPreprocessor
import torch
import torch.nn as nn
import os
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
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=Config.LEARNING_RATE, weight_decay = 1e-4)

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