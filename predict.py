import os
import json
import gspread
import pandas as pd
import numpy as np
import joblib
import logging
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.metrics.pairwise import cosine_similarity
import time

start_time = time.time()

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # Load secrets for local testing
# with open("secrets.json", "r") as file:
#     secrets = json.load(file)

# GOOGLE_SHEETS_NAME = secrets["GOOGLE_SHEETS_NAME"]
# GOOGLE_SHEETS_CLEAN = secrets["GOOGLE_SHEETS_CLEAN"]
# creds_json = secrets["CREDENTIALS_FILE"]

# Load secrets for production
GOOGLE_SHEETS_NAME = os.getenv("GOOGLE_SHEETS_NAME")
GOOGLE_SHEETS_CLEAN = os.getenv("GOOGLE_SHEETS_CLEAN")
CREDENTIALS_FILE = os.getenv("GCP_CREDENTIALS_JSON")
if CREDENTIALS_FILE:
    creds_json = json.loads(CREDENTIALS_FILE)
else:
    raise ValueError("Missing Google Sheets API credentials. Set GCP_CREDENTIALS as a GitHub Secret.")

cred_time = time.time()
load_cred_time = cred_time - start_time
print(f"Loading credentials time: {load_cred_time:.4f} seconds")

# Load trained model and preprocessing tools
rf_model = joblib.load("model/random_forest_model.pkl")
tfidf_vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

model_time = time.time()
load_model_time = model_time - cred_time
print(f"Loading models time: {load_model_time:.4f} seconds")

def connect_google_sheets():
    """Authenticate with Google Sheets API."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
    client = gspread.authorize(creds)
    return client

def read_google_sheet(sheet_name):
    """Reads data from Google Sheets and returns a Pandas DataFrame."""
    try:
        client = connect_google_sheets()
        sheet = client.open(sheet_name).sheet1
        data = sheet.get_all_records()
        logging.info("Successfully loaded data from Google Sheets.")
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"Error reading Google Sheets: {e}")
        raise

def preprocess_data(df):
    """Preprocesses input data for prediction."""
    df['activity_name'] = df['activity_name'].astype(str)

    df["commute_keyword"] = (cosine_similarity(
        tfidf_vectorizer.transform(df["activity_name"].fillna("")),
        tfidf_vectorizer.transform(["commute"])
    ).max(axis=1) >= 0.7).astype(int)

    # Ensure all new values exist in the trained label encoder
    df["workout_type"] = df["workout_type"].astype(str)

    # Use .get() to map unknown labels to a default (-1)
    df["workout_type_encoded"] = df["workout_type"].map(lambda x: label_encoders["workout_type"].transform([x])[0] 
                                                        if x in label_encoders["workout_type"].classes_ 
                                                        else -1)

    # Median distance feature engineering
    df["estimated_commute_distance"] = df.groupby("athlete_name")["distance"].transform("median")
    threshold = 0.2  
    df["commute_distance_flag"] = ((df["distance"] >= df["estimated_commute_distance"] * (1 - threshold)) &
                                (df["distance"] <= df["estimated_commute_distance"] * (1 + threshold))).astype(int)

    X = np.hstack((
        scaler.transform(df[["distance", "moving_time", "elapsed_time", "total_elevation_gain", "hour", "day"]]),
        tfidf_vectorizer.transform(df["activity_name"].fillna("")).toarray(),
        df[["commute_keyword", "workout_type_encoded", "commute_distance_flag"]].values
    ))

    return df, X

def predict(df, X):
    """Runs the trained model to make predictions."""
    df["is_commute"] = rf_model.predict(X)
    return df

def save_to_google_sheets(df, sheet_name):
    """Writes the predictions back to Google Sheets."""
    try:
        client = connect_google_sheets()
        sheet = client.open(sheet_name).sheet1
        sheet.clear()
        sheet.update([df.columns.values.tolist()] + df.values.tolist())
        logging.info("Predictions successfully saved to Google Sheets.")
    except Exception as e:
        logging.error(f"Error saving to Google Sheets: {e}")
        raise

def main():
    df = read_google_sheet(GOOGLE_SHEETS_NAME)
    gsheet_time = time.time()
    load_gsheet_time = gsheet_time - model_time
    print(f"Loading data time: {load_gsheet_time:.4f} seconds")

    df_original = df.copy()    
    df_processed, X_new = preprocess_data(df)
    df_final = predict(df_original, X_new)
    df_final['carbon_saving'] = df['distance'] * 0.18/1000
    save_to_google_sheets(df_final, GOOGLE_SHEETS_CLEAN)
    pred_time = time.time()
    pred_commute_time = pred_time - gsheet_time
    print(f"Predicting commute time: {pred_commute_time:.4f} seconds")
    print(f"Total runtime: {pred_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
