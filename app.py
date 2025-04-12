# Step 1: Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Step 2: Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Step 3: Load the trained LSTM model
model = load_model('lstm_rnn_imdb.h5')
max_len = 500  # Length used during training

# Step 4: Helper Functions
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

# Step 5: Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Analyzer (LSTM)", layout="centered")

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.subheader("Using LSTM Recurrent Neural Network (RNN)")
st.write("This app classifies your movie review as **Positive** or **Negative** sentiment based on an LSTM model.")

# Review input
user_input = st.text_area("📝 Enter a movie review:", height=150, placeholder="E.g., 'This movie was amazing and full of emotion.'")

# Classification
if st.button("🚀 Classify Sentiment"):
    if user_input.strip():
        sentiment, score = predict_sentiment(user_input)
        
        if sentiment == 'Positive':
            st.markdown(f"<h3 style='color:green;'>✅ Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:red;'>❌ Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
        
        st.markdown(f"<h4 style='color:gray;'>📊 Confidence Score: {score:.2f}</h4>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a valid review before classifying.")
else:
    st.info("👈 Type your review above and click 'Classify Sentiment'")

# Footer / Designer Credit
st.markdown("<hr style='margin-top:2rem;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>🎨 Designed by <strong>Prudhvi Raju</strong></p>", unsafe_allow_html=True)
