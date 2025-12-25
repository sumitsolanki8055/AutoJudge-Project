import streamlit as st
import joblib
import os

# --- 1. Smart Model Loader ---
@st.cache_resource
def load_models():
    # Define possible paths (checking 'models' folder first, then root)
    base_dir = os.path.dirname(__file__)
    locations = [
        os.path.join(base_dir, 'models'),
        base_dir
    ]
    
    def find_and_load(filename):
        for loc in locations:
            path = os.path.join(loc, filename)
            # Check if it exists AND is a file (not a folder)
            if os.path.exists(path) and os.path.isfile(path):
                try:
                    return joblib.load(path)
                except:
                    continue
        return None

    # Load all 3 components
    clf = find_and_load('model_class.pkl')
    reg = find_and_load('model_score.pkl')
    vectorizer = find_and_load('tfidf.pkl')
    
    return clf, reg, vectorizer

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

# --- 4. Prediction Logic (Updated to Fix Crash) ---
if st.button("🚀 Predict Difficulty", use_container_width=True):
    # CRITICAL FIX: Check ALL models before running
    if clf is None:
        st.error("❌ Error: 'model_class.pkl' is missing. Please check your 'models' folder.")
    elif reg is None:
        st.error("❌ Error: 'model_score.pkl' is missing. Please check your 'models' folder.")
    elif vectorizer is None:
        st.error("❌ Error: 'tfidf.pkl' is missing. Please check your 'models' folder.")
    elif not desc_input:
        st.warning("Please enter at least a Problem Description.")
    else:
        # If we get here, everything is safe to run!
        combined_text = f"{desc_input} {inp_input} {out_input}"
        
        # 1. Vectorize
        text_vectorized = vectorizer.transform([combined_text])
        
        # 2. Predict (Safe now because we checked they exist)
        predicted_class = clf.predict(text_vectorized)[0]
        predicted_score = reg.predict(text_vectorized)[0]
        
        # 3. Display Results
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
