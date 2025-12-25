import streamlit as st
import joblib
import pandas as pd

# Cache models to prevent reloading on every interaction
@st.cache_resource
def get_models():
    try:
        clf = joblib.load('model_class.pkl')
        reg = joblib.load('model_score.pkl')
        vec = joblib.load('tfidf.pkl')
        return clf, reg, vec
    except FileNotFoundError:
        return None, None, None

clf, reg, vectorizer = get_models()

st.set_page_config(page_title="AutoJudge", page_icon="🤖", layout="centered")

st.title("🤖 AutoJudge: Difficulty Predictor")
st.write("Predict programming problem difficulty based on the problem statement.")

# Layout for input fields
col1, col2 = st.columns(2)

with col1:
    desc = st.text_area("Problem Description", height=200, 
                        placeholder="Paste the main task details here...")

with col2:
    inp_desc = st.text_area("Input Constraints", height=90, 
                            placeholder="e.g., The first line contains N...")
    out_desc = st.text_area("Output Description", height=90, 
                            placeholder="e.g., Print the result modulo 10^9...")

if st.button("Predict Difficulty", use_container_width=True):
    # Error handling
    if not clf:
        st.error("Model files not found. Please run 'train_model.py' first.")
        st.stop()
    
    if not desc:
        st.warning("Please provide a problem description.")
        st.stop()

    # Preprocess and Vectorize
    full_text = f"{desc} {inp_desc} {out_desc}"
    features = vectorizer.transform([full_text])
    
    # Predict
    pred_class = clf.predict(features)[0]
    pred_score = reg.predict(features)[0]
    
    # Visuals
    st.divider()
    
    c1, c2 = st.columns(2)
    color_map = {"Easy": "green", "Medium": "orange", "Hard": "red"}
    text_color = color_map.get(pred_class, "blue")

    with c1:
        st.subheader("Predicted Class")
        st.markdown(f"<h1 style='color:{text_color}'>{pred_class}</h1>", unsafe_allow_html=True)

    with c2:
        st.subheader("Difficulty Score")
        st.metric(label="0 - 100 Scale", value=f"{pred_score:.1f}")
        st.progress(min(pred_score / 100, 1.0))
