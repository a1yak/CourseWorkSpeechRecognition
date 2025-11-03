"""
Sentiment Analysis System for Text and Speech
Trains a model on IMDB dataset and creates a Gradio interface
"""


import gradio as gr
import speech_recognition as sr
import os

from Config import TOKENIZER_PATH, MODEL_PATH
from SentimentAnalyzer import SentimentAnalyzer


def transcribe_audio(audio_file):
    """Convert speech to text using SpeechRecognition"""
    if audio_file is None:
        return "No audio recorded"

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Error with speech recognition service: {e}"
    except Exception as e:
        return f"Error processing audio: {e}"


def create_gradio_interface(analyzer):
    """Create Gradio interface"""

    def analyze_text(text):
        result, score = analyzer.predict_text(text)
        return result

    def analyze_audio(audio):
        if audio is None:
            return "Please record audio first", "No text transcribed"

        # Transcribe audio
        text = transcribe_audio(audio)

        if text.startswith("Could not") or text.startswith("Error"):
            return text, text

        # Analyze sentiment
        result, score = analyzer.predict_text(text)

        return result, f"Transcribed text: {text}"

    # Create interface
    with gr.Blocks(title="Sentiment Analysis System") as demo:
        gr.Markdown("""
        # 🎭 Sentiment Analysis System
        Analyze sentiment from text or speech using a trained LSTM model on IMDB dataset.
        """)

        with gr.Tab("Text Analysis"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Enter text to analyze",
                        placeholder="Type your text here...",
                        lines=5
                    )
                    text_button = gr.Button("Analyze Text", variant="primary")

                with gr.Column():
                    text_output = gr.Textbox(label="Sentiment Result", lines=5)

            gr.Examples(
                examples=[
                    ["This movie was absolutely fantastic! I loved every moment of it."],
                    ["I'm not sure how I feel about this product. It has pros and cons."],
                    ["Terrible experience. Complete waste of time and money."],
                ],
                inputs=text_input
            )

        with gr.Tab("Speech Analysis"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Record your speech"
                    )
                    audio_button = gr.Button("Analyze Speech", variant="primary")

                with gr.Column():
                    audio_output = gr.Textbox(label="Sentiment Result", lines=5)
                    transcription_output = gr.Textbox(label="Transcription", lines=3)

        # Connect buttons
        text_button.click(fn=analyze_text, inputs=text_input, outputs=text_output)
        audio_button.click(
            fn=analyze_audio,
            inputs=audio_input,
            outputs=[audio_output, transcription_output]
        )

    return demo


def main():
    """Main training and interface launch"""
    analyzer = SentimentAnalyzer()

    # Check if model exists
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        print("Found existing model. Loading...")
        analyzer.load_saved_model()
        print("\nSkipping training. Remove model files to retrain.")
    else:
        print("No existing model found. Starting training...\n")

        # Load data
        train_data, val_data, test_data = analyzer.load_data()

        # Prepare data
        train_prep, val_prep, test_prep = analyzer.prepare_data(train_data, val_data, test_data)

        # Train model
        analyzer.train(train_prep, val_prep)

        # Evaluate
        analyzer.evaluate(test_prep)

        # Plot metrics
        analyzer.plot_training_history()

        # Save model
        analyzer.save_model()

    # Create and launch Gradio interface
    print("\n" + "=" * 50)
    print("Launching Gradio Interface...")
    print("=" * 50)

    demo = create_gradio_interface(analyzer)
    demo.launch(share=False)


if __name__ == "__main__":
    # Install required packages first:
    # pip install tensorflow scikit-learn datasets gradio SpeechRecognition pyaudio matplotlib
    main()