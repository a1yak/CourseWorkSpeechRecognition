import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from datasets import load_dataset
import pickle

from Config import MAX_LEN, MAX_WORDS, EMBEDDING_DIM, MODEL_PATH, BATCH_SIZE, EPOCHS, TOKENIZER_PATH


class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.history = None

    def load_data(self):
        """Load and prepare IMDB dataset"""
        print("Loading IMDB dataset...")
        dataset = load_dataset('imdb')

        # Extract texts and labels
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        test_texts = dataset['test']['text']
        test_labels = dataset['test']['label']

        # Split training data to create validation set
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.2, random_state=42
        )

        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")
        print(f"Test samples: {len(test_texts)}")

        return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

    def prepare_data(self, train_data, val_data, test_data):
        """Tokenize and pad sequences"""
        train_texts, train_labels = train_data
        val_texts, val_labels = val_data
        test_texts, test_labels = test_data

        print("\nTokenizing texts...")
        self.tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(train_texts)

        # Convert texts to sequences
        train_seq = self.tokenizer.texts_to_sequences(train_texts)
        val_seq = self.tokenizer.texts_to_sequences(val_texts)
        test_seq = self.tokenizer.texts_to_sequences(test_texts)

        # Pad sequences
        X_train = pad_sequences(train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
        X_val = pad_sequences(val_seq, maxlen=MAX_LEN, padding='post', truncating='post')
        X_test = pad_sequences(test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

        y_train = np.array(train_labels)
        y_val = np.array(val_labels)
        y_test = np.array(test_labels)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def build_model(self):
        """Build LSTM model with dropout to prevent overfitting"""
        print("\nBuilding model...")
        model = Sequential([
            Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.5),
            Bidirectional(LSTM(32)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print(model.summary())
        return model

    def train(self, train_data, val_data):
        """Train model with callbacks to prevent overfitting"""
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Callbacks to prevent overfitting
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )

        checkpoint = ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )

        print("\nTraining model...")
        self.model = self.build_model()

        self.history = self.model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )

        return self.history

    def evaluate(self, test_data):
        """Evaluate model on test set"""
        X_test, y_test = test_data

        print("\nEvaluating model...")
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['Negative', 'Positive']))

        return loss, accuracy

    def plot_training_history(self):
        """Plot training metrics to check for overfitting"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        print("\nTraining metrics saved to 'training_metrics.png'")
        plt.show()

        # Check for overfitting
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        gap = abs(final_train_acc - final_val_acc)

        print(f"\nOverfitting Analysis:")
        print(f"Final Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Accuracy Gap: {gap:.4f}")

        if gap < 0.05:
            print("✓ Model shows good generalization (gap < 5%)")
        elif gap < 0.10:
            print("⚠ Model shows slight overfitting (gap 5-10%)")
        else:
            print("✗ Model shows significant overfitting (gap > 10%)")

    def save_model(self):
        """Save model and tokenizer"""
        self.model.save(MODEL_PATH)
        with open(TOKENIZER_PATH, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"\nModel saved to {MODEL_PATH}")
        print(f"Tokenizer saved to {TOKENIZER_PATH}")

    def load_saved_model(self):
        """Load saved model and tokenizer"""
        self.model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print("Model and tokenizer loaded successfully")

    def predict_text(self, text):
        """Predict sentiment from text"""
        if not text or text.strip() == "":
            return "Please provide some text to analyze", 0.5

        # Preprocess
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

        # Predict
        prediction = self.model.predict(padded, verbose=0)[0][0]

        # Determine sentiment with confidence
        if prediction >= 0.6:
            sentiment = "Positive 😊"
        elif prediction <= 0.4:
            sentiment = "Negative 😞"
        else:
            sentiment = "Neutral 😐"

        confidence = max(prediction, 1 - prediction) * 100

        return f"{sentiment}\n\nConfidence: {confidence:.1f}%\nScore: {prediction:.4f}", prediction