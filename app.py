import streamlit as st
import joblib
import os

# --- 1. Load the Trained AI Models ---
@st.cache_resource
def load_models():
    try:
        # Get the current directory
        base_dir = os.path.dirname(__file__)
        
        # Construct paths to the 'models' folder
        path_clf = os.path.join(base_dir, 'models', 'model_class.pkl')
        path_reg = os.path.join(base_dir, 'models', 'model_score.pkl')
        path_vec = os.path.join(base_dir, 'models', 'tfidf.pkl')

        # Load the models
        clf = joblib.load(path_clf)
        reg = joblib.load(path_reg)
        vectorizer = joblib.load(path_vec)
        
        return clf, reg, vectorizer
    except Exception as e:
        # This error helps us debug if files are missing
        st.error(f"⚠️ Error loading models. Please check if 'models/model_class.pkl' exists in your repo. Error details: {e}")
        return None, None, None

clf, reg, vectorizer = load_models()

# --- 2. Configure the Page ---
st.set_page_config(page_title="AutoJudge AI", page_icon="🤖", layout="centered")

st.title("🤖 AutoJudge: Difficulty Predictor")
st.markdown("### Professional AI System")
st.markdown("Enter the problem details below to predict its difficulty using Machine Learning.")

# --- 3. Input Fields ---
col1, col2 = st.columns(2)

with col1:
    st.info("1. Problem Description")
    desc_input = st.text_area("Paste the main task here:", height=150, placeholder="e.g., Write a program to find the shortest path...")

with col2:
    st.info("2. Input Description")
    inp_input = st.text_area("Paste input constraints:", height=70, placeholder="e.g., The first line contains integer N...")
    
    st.info("3. Output Description")
    out_input = st.text_area("Paste expected output:", height=70, placeholder="e.g., Print the sum of the array...")

# --- 4. Prediction Logic ---
if st.button("🚀 Predict Difficulty", use_container_width=True):
    if not clf:
        st.error("Models are not loaded. Please check the file structure on GitHub.")
    elif not desc_input:
        st.warning("Please enter at least a Problem Description.")
    else:
        # Combine inputs
        combined_text = f"{desc_input} {inp_input} {out_input}"
        
        # Convert text to numbers
        text_vectorized = vectorizer.transform([combined_text])
        
        # Predict
        predicted_class = clf.predict(text_vectorized)[0]
        predicted_score = reg.predict(text_vectorized)[0]
        
        # Display Results
        st.divider()
        st.subheader("Analysis Results")
        
        color_map = {"Easy": "green", "Medium": "orange", "Hard": "red"}
        color = color_map.get(predicted_class, "blue")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.markdown("**Predicted Class**")
            st.markdown(f"<h1 style='color:{color}'>{predicted_class}</h1>", unsafe_allow_html=True)
            
        with result_col2:
            st.markdown("**Predicted Score (0-100)**")
            st.metric(label="Difficulty Score", value=f"{predicted_score:.1f}")
            st.progress(min(predicted_score / 100, 1.0))
            
        st.success("Prediction generated successfully.")
