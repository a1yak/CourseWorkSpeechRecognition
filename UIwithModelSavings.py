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
        MAX_SAMPLES = 50000
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