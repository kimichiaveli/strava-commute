import os
import requests
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import time
from datetime import datetime, timedelta

# Load secrets from environment variables
STRAVA_CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
GOOGLE_SHEETS_NAME = os.getenv("GOOGLE_SHEETS_NAME")
# Decode credentials path from GitHub Actions
CREDENTIALS_FILE = "service-account.json"

# Authenticate Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
client = gspread.authorize(creds)
sheet = client.open(GOOGLE_SHEETS_NAME).worksheet("tokens")  # Use a separate sheet for tokens

print("Secrets loaded from environment variables.")

# Strava API URL
CLUB_ID = os.getenv("STRAVA_CLUB_ID")
STRAVA_URL = f"https://www.strava.com/api/v3/clubs/{CLUB_ID}/activities?per_page=100"

# Function to refresh and store token
def refresh_access_token():
    try:
        # Read current refresh token from Google Sheets
        token_data = sheet.get("A2:C2")[0]
        current_refresh_token = token_data[1]  # Column B stores refresh_token

        # Request new tokens from Strava
        token_data = {
            "client_id": STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "refresh_token": current_refresh_token,
            "grant_type": "refresh_token",
        }
        response = requests.post("https://www.strava.com/oauth/token", data=token_data)

        if response.status_code == 200:
            new_tokens = response.json()
            access_token = new_tokens["access_token"]
            refresh_token = new_tokens["refresh_token"]  # Might be new
            expires_at = new_tokens["expires_at"]

            # **Store new tokens in Google Sheets**
            sheet.update("A1", [["access_token", "refresh_token", "expires_at"]])
            sheet.update("A2", [[access_token, refresh_token, expires_at]])

            print("Token refreshed successfully!")
            return access_token
        else:
            print(f"Failed to refresh token: {response.text}")
            return None

    except Exception as e:
        print(f"Error refreshing token: {e}")
        return None

# Function to get access token
def get_access_token():
    try:
        # Fetch token from Google Sheets
        token_data = sheet.get("A2:C2")[0]
        
        access_token, refresh_token, expires_at = token_data
        expires_at = int(expires_at)

        # Refresh token if expired
        if int(time.time()) >= expires_at:
            print("Access token expired. Refreshing...")
            return refresh_access_token()
        
        return access_token
    except Exception as e:
        print(f"Error retrieving token from Google Sheets: {e}")
        return refresh_access_token()  # Refresh if the sheet is empty

# Fetch club activities from Strava API
def fetch_strava_activities():
    access_token = get_access_token()
    if not access_token:
        print("Failed to obtain a valid access token. Exiting...")
        return []

    print(f"🔍 Debug: CLUB_ID={CLUB_ID}, Access Token={access_token[:6]}...")  # Partial token for security
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        response = requests.get(STRAVA_URL, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching Strava activities: {e}")
        return []

# Authenticate with Google Sheets
def authenticate_google_sheets():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        client = gspread.authorize(creds)
        return client.open(GOOGLE_SHEETS_NAME).sheet1
    except Exception as e:
        print(f"Google Sheets authentication failed: {e}")
        return None

# Read existing data from Google Sheets
def get_existing_data(sheet):
    try:
        data = sheet.get_all_records()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        print(f"Error reading existing data from Google Sheets: {e}")
        return pd.DataFrame()

# Append only new activities
def append_new_activities(sheet, activities):
    column_order = [
    "activity_name", "athlete_name", "distance",
    "moving_time", "elapsed_time", "total_elevation_gain",
    "workout_type", "hour", "day", "date", "start_date_local","record_date"
    ]
    
    if not sheet or not activities:
        print("No valid Google Sheets connection or activities data. Skipping update.")
        return

    try:
        required_columns = ["athlete", "name", "distance", "moving_time", "elapsed_time", "total_elevation_gain", "workout_type"]
        new_df = pd.DataFrame(activities)[required_columns]

        # Ensure required columns are present
        for col in required_columns:
            if col not in new_df.columns:
                new_df[col] = None

        new_df.rename(columns={"name": "activity_name"}, inplace=True)

        # Extract athlete details
        new_df["athlete_firstname"] = new_df["athlete"].apply(lambda x: x.get("firstname", "Unknown") if isinstance(x, dict) else "Unknown")
        new_df["athlete_lastname"] = new_df["athlete"].apply(lambda x: x.get("lastname", "Unknown") if isinstance(x, dict) else "Unknown")

        new_df.fillna(0, inplace=True)  # Replace NaN values with 0

        # Add timestamps
        utc_plus7 = datetime.utcnow() + timedelta(hours=7) # Assume activity starts in utc+7
        new_df["record_date"] = utc_plus7.strftime("%Y-%m-%dT%H:%M:%SZ")
        new_df["record_date"] = pd.to_datetime(new_df["record_date"])
        new_df["start_date_local"] = (new_df["record_date"] - pd.to_timedelta(new_df["elapsed_time"], unit="s")).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        new_df["date"] = new_df["start_date_local"].dt.strftime("%Y-%m-%d")  # Extract date (YYYY-MM-DD)
        new_df["day"] = new_df["start_date_local"].dt.weekday + 1  # Monday = 1, Sunday = 7
        new_df["hour"] = new_df["start_date_local"].dt.strftime("%H")  # Extract hour (HH)
        new_df["start_date_local"] = new_df["start_date_local"].dt.strftime("%Y-%m-%dT%H:%M:%SZ") # Format `start_date_local` as ISO 8601 for consistency

        # Load existing data
        existing_df = get_existing_data(sheet)

        # Convert all values to JSON-friendly types
        new_df = new_df.astype(str)
        existing_df = existing_df.astype(str)

        # **Merge athlete's name to avoid duplicate rows caused by number-only lastname
        new_df["athlete_name"] = new_df["athlete_firstname"] + " " + new_df["athlete_lastname"]
        new_df.drop(columns=["athlete", "athlete_firstname", "athlete_lastname"], inplace=True, errors="ignore")
        new_df_clean = new_df[column_order]

        # If sheet is empty, insert all activities
        if existing_df.empty:
            sheet.append_rows(new_df_clean.values.tolist(), value_input_option="RAW")
            print("All activities added.")
            return

        # Filter out duplicates
        key_columns = ["athlete_name", "activity_name", "distance"]
        merged_df = existing_df.merge(new_df_clean, on=key_columns, how="outer", indicator=True)

        # Replace original columns with new_df values (keep `_y`)
        for col in merged_df.columns:
            if col.endswith("_x") or col.endswith("_y"):
                base_col = col[:-2]  # Remove _x or _y suffix
                merged_df[base_col] = merged_df.get(base_col + "_y", merged_df.get(base_col + "_x"))
                merged_df.drop(columns=[col], inplace=True, errors="ignore")

        new_entries = merged_df[merged_df["_merge"] == "right_only"].drop(columns=["_merge"])[column_order]

        # Convert all values to JSON-friendly types
        new_entries = new_entries.astype(str)

        if not new_entries.empty:
            sheet.append_rows(new_entries.values.tolist(), value_input_option="RAW")
            print(f"Added {len(new_entries)} new activities.")
        else:
            print("No new activities to add.")
    except Exception as e:
        print(f"Error processing new activities: {e}")

# Main Execution
def main():
    print("Fetching Strava activities...")
    activities = fetch_strava_activities()

    if not activities:
        print("No data retrieved from Strava. Exiting...")
        return

    sheet = authenticate_google_sheets()
    if not sheet:
        print("Skipping update due to authentication failure.")
        return

    append_new_activities(sheet, activities)

if __name__ == "__main__":
    main()
