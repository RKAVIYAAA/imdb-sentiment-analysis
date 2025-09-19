import torch
import gradio as gr
from transformers import pipeline
import matplotlib.pyplot as plt

# Load Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Store reviews and sentiment
review_list = []

def predict_sentiment(review_text):
    result = sentiment_pipeline(review_text)
    sentiment = result[0]['label']
    # Append the review and sentiment to the list
    review_list.append({
        "Review": review_text,
        "Sentiment": sentiment,
        "Emoji": "😊" if sentiment == "POSITIVE" else "😠"
    })
    return sentiment

def plot_distribution():
    pos = sum(1 for r in review_list if r["Sentiment"] == "POSITIVE")
    neg = sum(1 for r in review_list if r["Sentiment"] == "NEGATIVE")
    neu = len(review_list) - pos - neg
    total = pos + neg + neu
    fig, ax = plt.subplots()
    if total == 0:
        ax.text(0.5, 0.5, "No data available", horizontalalignment='center', verticalalignment='center', fontsize=14)
    else:
        colors = ["#4CAF50", "#F44336", "#FFC107"]
        ax.pie([pos, neg, neu], labels=["Positive 😊", "Negative 😠", "Neutral 😐"], colors=colors, autopct='%1.1f%%')
    ax.set_title("Sentiment Distribution")
    return fig

def get_review_table():
    # Return recent reviews as a table
    return [[r["Review"], r["Sentiment"], r["Emoji"]] for r in review_list[-6:]]

custom_css = """
body {background: linear-gradient(90deg, #30343F 70%, #FFFCF9 100%);}
.gradio-container {color: #F8CB2E;}
h1,h2,h3 {color:#F8CB2E;}
.gr-button {background-color:#30343F; color:#F8CB2E; font-size:18px;}
"""

with gr.Blocks(theme="dark", css=custom_css) as demo:
    gr.Markdown("# 🎬 **CineSense Movie Review Analytics Dashboard**")
    gr.Markdown("Monitor your movie review sentiment analysis 🔍✨")
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="📥 Your Movie Review", placeholder="E.g. This film was amazing!", lines=2)
            submit_btn = gr.Button("Submit")
            out = gr.Textbox(label="🔎 Sentiment")
        with gr.Column():
            chart = gr.Plot(plot_distribution, label="Sentiment Distribution")
    with gr.Row():
        gr.Markdown("### 📝 Recent Reviews")
        review_table = gr.Dataframe(get_review_table, headers=["Review", "Sentiment", "Emoji"], label="Recent Reviews")
    submit_btn.click(predict_sentiment, inputs=inp, outputs=out).then(
        lambda: None, None, chart
    ).then(
        lambda: None, None, review_table
    )

if __name__ == "__main__":
    demo.launch(share=True)

    demo.launch(share=True)

