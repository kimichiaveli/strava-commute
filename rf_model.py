
import json
import gspread
import logging
import pandas as pd
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import joblib

# Setup Logging for Debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load secrets from secrets.json
with open("secrets.json", "r") as file:
    secrets = json.load(file)

GOOGLE_SHEETS_MODEL = secrets["GOOGLE_SHEETS_MODEL"]
CREDENTIALS_FILE = secrets["CREDENTIALS_FILE"]

def load_commute_keywords(filepath="commute.txt"):
    """Reads commute-related keywords from a text file."""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            keywords = [line.strip() for line in file.readlines() if line.strip()]
        return keywords
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return []

# Load commute-related keywords from file
COMMUTE_KEYWORDS = load_commute_keywords()

# Authenticate and Read Data from Google Sheets
def read_google_sheet(sheet_name):
    """Reads data from Google Sheets and returns it as a Pandas DataFrame."""
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(CREDENTIALS_FILE, scope)
        client = gspread.authorize(creds)
        sheet = client.open(sheet_name).sheet1
        data = sheet.get_all_records()
        logging.info("Successfully loaded data from Google Sheets.")
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"Error reading Google Sheet: {e}")
        raise

# Feature Engineering Functions
def preprocess_text_features(df):
    """Vectorizes `activity_name` using TF-IDF and computes cosine similarity with commute-related words."""
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

    # Fit TF-IDF on `activity_name` and commute keywords
    all_texts = df["activity_name"].fillna("").tolist() + COMMUTE_KEYWORDS
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

    # Split matrix: first part for activities, last part for commute keywords
    activity_vectors = tfidf_matrix[:len(df)]
    keyword_vectors = tfidf_matrix[len(df):]

    # Compute Cosine Similarity
    cosine_sim = cosine_similarity(activity_vectors, keyword_vectors)
    max_sim_scores = cosine_sim.max(axis=1)

    # Convert to binary feature
    df["commute_keyword"] = (max_sim_scores >= 0.5).astype(int)
    logging.info("Text features processed with TF-IDF and cosine similarity.")

    return df, tfidf_vectorizer

def encode_categorical_features(df):
    """Encodes categorical columns (workout_type) using Label Encoding."""
    label_encoders = {}
    le = LabelEncoder()
    df["workout_type"] = le.fit_transform(df["workout_type"].astype(str))
    label_encoders["workout_type"] = le
    logging.info("Categorical features encoded.")
    return df, label_encoders

def normalize_numerical_features(df, feature_cols):
    """Normalizes numerical columns using StandardScaler."""
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(df[feature_cols])
    logging.info("Numerical features normalized.")
    return X_num_scaled, scaler

# Load and Preprocess Data
df = read_google_sheet(GOOGLE_SHEETS_MODEL)

# Process text features (TF-IDF + Cosine Similarity)
df, tfidf_vectorizer = preprocess_text_features(df)

# Encode categorical features
df, label_encoders = encode_categorical_features(df)

# Normalize numerical features
numerical_cols = ["distance", "moving_time", "elapsed_time", "total_elevation_gain", "hour", "day"]
X_num_scaled, scaler = normalize_numerical_features(df, numerical_cols)

# Extract categorical & binary features
X_other = df[["commute_keyword", "workout_type"]].values

# Vectorize Activity Name Using TF-IDF
activity_name_tfidf = tfidf_vectorizer.transform(df["activity_name"].fillna("")).toarray()

# Combine Features
X = np.hstack((X_num_scaled, activity_name_tfidf, X_other))
y = df["is_commute"].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # ** SMOTE ONLY **
# smote = SMOTE(sampling_strategy=0.4, random_state=42)  # Only upsample non-commutes to 40%
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# ** SMOTE Tomek **
smote_tomek = SMOTETomek(sampling_strategy=0.4, random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
logging.info("Random Forest model trained.")

# Evaluate Model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {accuracy:.4f}")
logging.info("\n" + classification_report(y_test, y_pred))

# Save Model & Preprocessing Tools
joblib.dump(rf_model, "model/random_forest_model.pkl")
joblib.dump(tfidf_vectorizer, "model/tfidf_vectorizer.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")

logging.info("Model training complete! Model saved.")
