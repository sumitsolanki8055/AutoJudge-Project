# 🤖 AutoJudge: Difficulty Predictor

An AI-powered application that predicts programming problem difficulty using **text analysis** (problem descriptions) and **machine learning**. Built with Scikit-Learn (Random Forest) and Streamlit.

## 🚀 Live Demo
[Link to Live App](https://autojudge-project-fayud3ugxmanel6ejbfaqz.streamlit.app/)

## 🧠 How It Works
This project uses a **Dual-Model Approach**:
1.  **NLP Processor (TF-IDF):** Converts raw problem text into numerical features the computer can understand.
2.  **Classification Model:** Predicts the categorical difficulty level (**Easy, Medium, Hard**).
3.  **Regression Model:** Predicts a precise numerical complexity score (**0-100**).

## 🛠️ Tech Stack
* **Python**
* **Scikit-Learn** (Machine Learning)
* **Streamlit** (Web UI)
* **Pandas & NumPy** (Data Processing)
* **BeautifulSoup** (Web Scraping)

## 📂 Project Structure
    task_dataset.json       # Scraped and generated dataset
    app.py                  # Main Streamlit application
    train_model.py          # Script to train and save AI models
    generate_data.py        # Generates synthetic training data
    scraper.py              # Scrapes problems from Project Euler
    model_class.pkl         # Trained Classification Model
    requirements.txt        # List of dependencies

## 💻 How to Run Locally

1. **Clone the repository**
   git clone https://github.com/sumitsolanki8055/AutoJudge-Project.git
   cd AutoJudge-Project

2. **Install dependencies**
   pip install -r requirements.txt

3. **Run the app**
   streamlit run app.py

## 📊 Dataset
The model was trained on a hybrid dataset containing real-world problems scraped from **Project Euler** and synthetically generated tasks. The data was processed and cleaned using the `generate_data.py` script.

---
*Created by Sumit Solanki*
