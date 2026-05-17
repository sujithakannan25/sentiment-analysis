import streamlit as st
import joblib
import os
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)

# Page config
st.set_page_config(
    page_title="💬 Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(BASE_DIR, 'src', 'sentiment_model.pkl'))
    tfidf = joblib.load(os.path.join(BASE_DIR, 'src', 'tfidf_vectorizer.pkl'))
    return model, tfidf

model, tfidf = load_model()

# Clean text function
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split()
                     if word not in stop_words])
    return text

# Header
st.title("💬 Sentiment Analyzer")
st.markdown("**Built by Sujitha | AI/ML Portfolio Project**")
st.markdown("---")

# Input
st.subheader("📝 Enter Your Review")
user_input = st.text_area(
    "Type any product review here...",
    placeholder="Example: This product is amazing! I love it.",
    height=150
)

st.markdown("---")

# Predict
if st.button("🔍 Analyze Sentiment", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review!")
    else:
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        if prediction == 1:
            st.success("### 😊 Positive Sentiment!")
            st.metric("Confidence", f"{probability[1]*100:.1f}%")
        else:
            st.error("### 😞 Negative Sentiment!")
            st.metric("Confidence", f"{probability[0]*100:.1f}%")

st.markdown("---")
st.caption("🤖 Powered by Logistic Regression + TF-IDF | Dataset: Amazon Reviews")