import streamlit as st
import joblib

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("ai_detector_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Page config
st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🤖",
    layout="centered"
)

# Title and description
st.title("🤖 AI vs Human Text Detector")
st.markdown("Enter some text below to check if it was written by **AI** or a **Human**.")

# Text input
text_input = st.text_area(
    "Paste your text here:",
    height=200,
    placeholder="Enter the text you want to analyze..."
)

# Predict button
if st.button("🔍 Analyze Text", type="primary"):
    if text_input.strip():
        # Make prediction
        text_vec = vectorizer.transform([text_input])
        prediction = model.predict(text_vec)[0]
        probs = model.predict_proba(text_vec)[0]
        confidence = probs[int(prediction)]
        
        # Display results
        label = "🤖 AI Generated" if prediction == 1.0 else "✍️ Human Written"
        
        st.markdown("---")
        st.subheader("Result:")
        
        if prediction == 1.0:
            st.error(f"**{label}**")
        else:
            st.success(f"**{label}**")
        
        # Confidence meter
        st.metric("Confidence", f"{confidence:.1%}")
        
        # Show probability breakdown
        st.markdown("##### Probability Breakdown:")
        col1, col2 = st.columns(2)
        col1.metric("Human", f"{probs[0]:.1%}")
        col2.metric("AI", f"{probs[1]:.1%}")
        
    else:
        st.warning("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Model: Logistic Regression + TF-IDF")