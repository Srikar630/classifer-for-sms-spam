import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")

st.title("📩 Spam SMS Classifier")
text = st.text_area("Enter a message:")

if st.button("Predict"):
    if text.strip():
        prob = model.predict_proba([text])[0][1]
        if prob >= 0.5:
            st.error(f"🚨 Spam ({prob:.1%} confidence)")
        else:
            st.success(f"✅ Not Spam ({1 - prob:.1%} confidence)")
    else:
        st.warning("Please enter a message.")
