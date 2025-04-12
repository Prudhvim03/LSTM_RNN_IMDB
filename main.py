

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
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # 2 = unknown
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input, verbose=0)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Step 5: Streamlit App UI
st.set_page_config(page_title="IMDB Sentiment Analyzer (LSTM)", layout="centered")
st.title('🎬 IMDB Movie Review Sentiment Analysis (LSTM RNN)')
st.write('This app uses an LSTM RNN to determine if a movie review is **positive** or **negative**.')

# User input
user_input = st.text_area('📝 Enter a movie review:', height=150, placeholder='Type your review here...')

if st.button('Classify'):
    if user_input.strip():
        sentiment, score = predict_sentiment(user_input)
        st.markdown(f"### Sentiment: **{sentiment}**")
        st.markdown(f"### Confidence Score: `{score:.2f}`")
    else:
        st.warning("⚠️ Please enter a valid review before classifying.")
else:
    st.info("👈 Enter a review and click 'Classify' to begin.")
