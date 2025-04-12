# 📦 Step 1: Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# 📖 Step 2: Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# 💾 Step 3: Load the trained LSTM model
model = load_model('lstm_rnn_imdb.h5')
max_len = 500

# 🧠 Step 4: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    return sequence.pad_sequences([encoded_review], maxlen=max_len)

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input, verbose=0)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# 🎨 Step 5: Streamlit UI Setup
st.set_page_config(page_title="🎬 IMDB LSTM Sentiment Analyzer", layout="centered")

# 🖌️ Custom CSS: Movie-Themed Background and Typography
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600&display=swap');

    html, body {
        background-image: url("https://images.unsplash.com/photo-1601991050721-cd3a0ee9b93e");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }

    .main {
        background-color: rgba(0, 0, 0, 0.75);
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 0 20px rgba(255,255,255,0.2);
        max-width: 700px;
        margin: auto;
        margin-top: 50px;
        font-family: 'Cinzel', serif;
    }

    .footer {
        text-align: center;
        color: #bbbbbb;
        font-size: 0.85rem;
        margin-top: 50px;
        font-family: 'Courier New', monospace;
    }

    textarea {
        background-color: #fdfdfd !important;
        color: black !important;
        border: 1px solid #ddd !important;
    }

    .stButton>button {
        background-color: #e50914;
        color: white;
        border-radius: 6px;
        padding: 0.5em 1.5em;
        font-weight: bold;
        border: none;
    }

    .stButton>button:hover {
        background-color: #b20710;
    }

    .stSuccess, .stInfo {
        font-size: 1.1rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 🎬 Title and UI Elements
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("## 🎥 IMDB Sentiment Analyzer", unsafe_allow_html=True)
st.markdown("### Predict movie review sentiment using LSTM neural network")
st.write("Enter a review of a movie and let the model predict whether it's **positive** or **negative**.")

# ✍️ Review Input
user_input = st.text_area("📝 Your Movie Review", height=150, placeholder="E.g., 'This film was a masterpiece of emotion and suspense.'")

# 🔘 Classification Button
if st.button("🎞️ Classify Review"):
    if user_input.strip():
        sentiment, score = predict_sentiment(user_input)
        st.success(f"🎯 Sentiment: {sentiment}")
        st.info(f"📊 Confidence Score: {score:.2f}")
    else:
        st.warning("⚠️ Please write a valid review before submitting.")
else:
    st.caption("👈 Type your review above and click 'Classify Review'")

# ✨ Footer
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>🎨 Designed with 💡 by <strong>Your Name</strong></div>", unsafe_allow_html=True)
