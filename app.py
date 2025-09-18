import torch
import gradio as gr
from transformers import pipeline

# Load Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def predict_sentiment(review_text):
    result = sentiment_pipeline(review_text)
    return result[0]['label']

custom_css = """
/* Background and font */
body {
    background-color: #f0f8ff;  /* Light blue background */
    font-family: 'Arial', sans-serif;
}

/* Customize Gradio buttons */
.gr-button {
    background-color: #ff4500; /* Orange-red */
    color: white;
    font-weight: bold;
    font-size: 16px;
    border-radius: 10px;
    border: none;
    padding: 10px 25px;
}

/* Input and output boxes */
.gr-input textarea, .gr-output textarea {
    font-size: 18px;
    color: #333333; /* Dark grey text */
    background-color: #fff8dc; /* Light beige background */
    border-radius: 8px;
    border: 2px solid #ff4500; /* Same accent color */
}

/* Title style */
h1, h2, h3 {
    font-family: 'Verdana', Geneva, Tahoma, sans-serif;
    color: #004080;  /* Navy blue */
}
"""

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Your Movie Review", placeholder="E.g. This film was amazing!", lines=2),
    outputs=gr.Textbox(label="Sentiment"),
    title="🎬 IMDb Review Sentiment Analyzer",
    description="⭐ Enter your review and discover if it's POSITIVE or NEGATIVE!",
    css=custom_css,
    allow_flagging="never"  # Disable flagging to avoid .gradio folder creation
)

if __name__ == "__main__":
    iface.launch()

