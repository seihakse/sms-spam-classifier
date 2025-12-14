import streamlit as st
import pickle
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --------------------
# Setup
# --------------------
ps = PorterStemmer()

@st.cache_resource
def load_nltk_data():
    nltk.download('stopwords')

load_nltk_data()

STOP_WORDS = set(stopwords.words('english'))

# --------------------
# Text Preprocessing
# --------------------
def transform_text(text):
    text = text.lower()
    tokens = wordpunct_tokenize(text)

    # Keep alphanumeric tokens
    tokens = [word for word in tokens if word.isalnum()]

    # Remove stopwords
    tokens = [word for word in tokens if word not in STOP_WORDS]

    # Stemming
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)

# --------------------
# Load Model & Vectorizer
# --------------------
@st.cache_resource
def load_model():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_model()

# --------------------
# Streamlit UI
# --------------------
st.title("📩 Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        with st.spinner("Analyzing message..."):
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 IT'S A SPAM MESSAGE")
        else:
            st.success("✅ IT'S A HAM MESSAGE")
