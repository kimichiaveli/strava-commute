import requests
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import time
from datetime import datetime

# Strava API Credentials
STRAVA_CLIENT_ID = "140601"
STRAVA_CLIENT_SECRET = "fdfdfa731116c0808761787a82b07f2f18b78de1"
STRAVA_REFRESH_TOKEN = "5ac1019a1d6748cabb237a248aae1d494c5f1558"
TOKEN_FILE = "strava-commute/strava_token.json"  # Store the latest token here

# Google Sheets Configuration
GOOGLE_SHEETS_NAME = "commute_raw"
CREDENTIALS_FILE = "cred/service-account.json"

# Strava API URL
CLUB_ID = "1118221"
STRAVA_URL = f"https://www.strava.com/api/v3/clubs/{CLUB_ID}/activities?per_page=100"

# Function to refresh Strava access token
def refresh_access_token():
    token_data = {
        "client_id": STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "refresh_token": STRAVA_REFRESH_TOKEN,
        "grant_type": "refresh_token",
    }
    try:
        response = requests.post("https://www.strava.com/oauth/token", data=token_data, timeout=10)
        response.raise_for_status()
        new_tokens = response.json()

        # Save new tokens
        with open(TOKEN_FILE, "w") as f:
            json.dump({
                "access_token": new_tokens.get("access_token"),
                "refresh_token": new_tokens.get("refresh_token"),
                "expires_at": new_tokens.get("expires_at")
            }, f)

        return new_tokens.get("access_token")
    except requests.RequestException as e:
        print(f"Error refreshing Strava token: {e}")
        return None

# Get valid access token (refresh if needed)
def get_access_token():
    try:
        with open(TOKEN_FILE, "r") as f:
            tokens = json.load(f)

        access_token = tokens.get("access_token")
        expires_at = tokens.get("expires_at")

        if not access_token or not expires_at:
            print("Invalid token file. Refreshing token...")
            return refresh_access_token()

        if time.time() >= expires_at:
            print("Access token expired. Refreshing...")
            return refresh_access_token()

        return access_token
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Token file missing or corrupted: {e}. Refreshing token...")
        return refresh_access_token()

# Fetch club activities from Strava API
def fetch_strava_activities():
    access_token = get_access_token()
    if not access_token:
        print("Failed to obtain a valid access token. Exiting...")
        return []

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
        new_df["start_date_local"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        new_df["date"] = datetime.now().strftime("%Y-%m-%d")
        new_df["day"] = datetime.now().weekday() + 1  # Monday = 1, Sunday = 7
        new_df["hour"] = datetime.now().strftime("%H")

        # Load existing data
        existing_df = get_existing_data(sheet)

        # Convert all values to JSON-friendly types
        new_df = new_df.astype(str)
        existing_df = existing_df.astype(str)

        # **Merge athlete's name to avoid duplicate rows caused by number-only lastname
        new_df["athlete_name"] = new_df["athlete_firstname"] + " " + new_df["athlete_lastname"]
        new_df.drop(columns=["athlete", "athlete_firstname", "athlete_lastname"], inplace=True, errors="ignore")

        # If sheet is empty, insert all activities
        if existing_df.empty:
            sheet.append_rows(new_df.values.tolist(), value_input_option="RAW")
            print("All activities added.")
            return

        # Filter out duplicates
        key_columns = ["athlete_name", "activity_name", "distance"]
        merged_df = existing_df.merge(new_df, on=key_columns, how="outer", indicator=True)

        # Replace original columns with new_df values (keep `_y`)
        for col in merged_df.columns:
            if col.endswith("_x") or col.endswith("_y"):
                base_col = col[:-2]  # Remove _x or _y suffix
                merged_df[base_col] = merged_df.get(base_col + "_y", merged_df.get(base_col + "_x"))
                merged_df.drop(columns=[col], inplace=True, errors="ignore")

        new_entries = merged_df[merged_df["_merge"] == "right_only"].drop(columns=["_merge"])[existing_df.columns]

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