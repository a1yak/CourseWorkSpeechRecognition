import warnings
import pandas as pd
import os
import gradio as gr

from Config import Config
from IntegratedSystem import SpeechSemanticAnalyzer

warnings.filterwarnings('ignore')

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

def load_twitter_sentiment_dataset(dataset_path):
    """
    Load Twitter sentiment dataset from Kaggle
    This dataset has: text_id, text, selected_text, sentiment
    """
    print(f"Loading Twitter sentiment dataset from: {dataset_path}")

    csv_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")

    print(f"Found {len(csv_files)} CSV file(s)")

    all_texts = []
    all_labels = []

    for csv_file in csv_files:
        print(f"Loading: {csv_file}")
        df = pd.read_csv(csv_file)

        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        print(df.head(3))

        # This dataset has specific columns
        if 'text' in df.columns and 'sentiment' in df.columns:
            # Remove rows with missing values
            df = df.dropna(subset=['text', 'sentiment'])

            texts = df['text'].astype(str).tolist()
            labels_raw = df['sentiment'].tolist()

            # Filter out any weird labels - keep only positive, negative, neutral
            valid_sentiments = {'positive', 'negative', 'neutral'}
            filtered_texts = []
            filtered_labels = []

            for text, label in zip(texts, labels_raw):
                label_lower = str(label).lower().strip()
                if label_lower in valid_sentiments:
                    filtered_texts.append(text)
                    filtered_labels.append(label_lower)

            all_texts.extend(filtered_texts)
            all_labels.extend(filtered_labels)

            print(f"Valid samples extracted: {len(filtered_texts)}")
        else:
            print(f"Warning: Expected columns 'text' and 'sentiment' not found")

    if not all_texts:
        raise ValueError("No valid data found in dataset")

    # Convert labels to numeric indices
    # Force order: negative(0), neutral(1), positive(2) for consistency
    unique_labels = ['negative', 'neutral', 'positive']
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_to_idx[label] for label in all_labels]

    print(f"\n✅ Dataset loaded successfully!")
    print(f"Total samples: {len(all_texts)}")
    print(f"Categories: {unique_labels}")

    # Show distribution
    label_counts = Counter(all_labels)
    for label in unique_labels:
        count = label_counts.get(label, 0)
        percentage = (count / len(all_labels)) * 100
        print(f"  {label}: {count} samples ({percentage:.1f}%)")

    return all_texts, numeric_labels, unique_labels


def load_custom_csv(csv_path, text_column='text', label_column='sentiment'):
    """
    Load custom CSV with text and sentiment labels

    Args:
        csv_path: path to your CSV file
        text_column: name of column with text
        label_column: name of column with labels (e.g., 'positive', 'negative', 'neutral')

    Returns:
        texts, numeric_labels, categories
    """
    print(f"Loading custom dataset from: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")

    # Extract texts and labels
    texts = df[text_column].astype(str).tolist()
    labels_raw = df[label_column].tolist()

    # Convert to numeric
    unique_labels = sorted(list(set(labels_raw)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_to_idx[label] for label in labels_raw]

    print(f"Categories: {unique_labels}")
    print(f"Distribution: {pd.Series(labels_raw).value_counts().to_dict()}")

    return texts, numeric_labels, unique_labels


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def train_mode():
    """Training mode - train and save model"""
    print("\n" + "="*60)
    print("TRAINING MODE")
    print("="*60 + "\n")

    # ========================================================================
    # DATASET SELECTION - Choose one:
    # ========================================================================

    DATASET_SOURCE = "twitter"  # Options: "twitter", "custom_csv", "example"

    if DATASET_SOURCE == "twitter":
        # Twitter sentiment dataset (recommended)
        try:
            import kagglehub
            print("Downloading Twitter sentiment dataset...")
            dataset_path = kagglehub.dataset_download("yasserh/twitter-tweets-sentiment-dataset")
            texts, labels, categories = load_twitter_sentiment_dataset(dataset_path)

            # Optionally limit for faster training (remove this for full dataset)
            MAX_SAMPLES = 10000  # Use 10k samples, or set to None for all
            if MAX_SAMPLES and len(texts) > MAX_SAMPLES:
                print(f"\n⚡ Limiting to {MAX_SAMPLES} samples for faster training...")
                # Shuffle first to get random sample
                combined = list(zip(texts, labels))
                np.random.shuffle(combined)
                texts, labels = zip(*combined[:MAX_SAMPLES])
                texts, labels = list(texts), list(labels)

        except Exception as e:
            print(f"Error loading Twitter dataset: {e}")
            print("Falling back to example data...")
            DATASET_SOURCE = "example"

    elif DATASET_SOURCE == "custom_csv":
        # Option: Load your own CSV file
        texts, labels, categories = load_custom_csv(
            csv_path="path/to/your/dataset.csv",
            text_column="text",
            label_column="sentiment"
        )

    if DATASET_SOURCE == "example":
        # Fallback: Use small example data
        print("Using example data...")
        categories = ['negative', 'neutral', 'positive']
        texts = [
            "this is terrible I hate it",
            "it's okay nothing special",
            "this is amazing I love it",
            "worst experience ever",
            "average quality",
            "absolutely wonderful",
        ] * 50
        labels = [0, 1, 2, 0, 1, 2] * 50

    # Train model
    analyzer = SpeechSemanticAnalyzer(categories)
    analyzer.train_classifier(texts, labels)

    # Save model
    analyzer.save_model()

    print("\n✅ Training complete! Model saved to", Config.MODEL_DIR)
    print(f"📈 Check training plots in: {Config.PLOTS_DIR}")
    print("\n🚀 Now run in inference mode to use the trained model.")


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