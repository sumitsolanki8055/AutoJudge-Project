import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv("programming_problems.csv")

# 2. Data Preprocessing (Combine text columns)
# [cite_start]This satisfies the PDF requirement to "Combine all text fields" [cite: 50]
print("Preprocessing data...")
df['combined_text'] = df['title'] + " " + df['description'] + " " + df['input_description'] + " " + df['output_description']

X = df['combined_text']
y_class = df['problem_class']  # Target for Classification (Easy/Medium/Hard)
y_score = df['problem_score']  # Target for Regression (Numerical Score)

# 3. Feature Extraction (TF-IDF)
# [cite_start]This satisfies the PDF requirement for "Feature Extraction" and "TF-IDF vectors" [cite: 31, 56]
print("Vectorizing text...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_vectorized = vectorizer.fit_transform(X)

# 4. Split Data (Train vs Test)
X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X_vectorized, y_class, y_score, test_size=0.2, random_state=42
)

# 5. Train Classification Model (Random Forest)
# [cite_start]This satisfies "Model 1: Classification" [cite: 33]
print("Training Classification Model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_class_train)

# Evaluate Classification
y_pred_class = clf.predict(X_test)
acc = accuracy_score(y_class_test, y_pred_class)
print(f"✅ Classification Accuracy: {acc * 100:.2f}%")
print("Confusion Matrix Report:")
print(classification_report(y_class_test, y_pred_class))

# 6. Train Regression Model (Random Forest Regressor)
# [cite_start]This satisfies "Model 2: Regression" [cite: 35]
print("Training Regression Model...")
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_score_train)

# Evaluate Regression
y_pred_score = reg.predict(X_test)
mae = mean_absolute_error(y_score_test, y_pred_score)
print(f"✅ Regression MAE (Average Error): {mae:.2f} points")

# 7. Save the Models (So app.py can use them)
print("Saving models...")
joblib.dump(clf, 'model_class.pkl')    # The Classifier
joblib.dump(reg, 'model_score.pkl')    # The Regressor
joblib.dump(vectorizer, 'tfidf.pkl')   # The Translator (Text -> Numbers)

print("--------------------------------------------------")
print("SUCCESS: AI Models trained and saved to .pkl files!")