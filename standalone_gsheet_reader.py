"""
Standalone Google Sheets Reader
Reads credentials from .streamlit/secrets.toml and provides functions to read Google Sheets.
"""

import tomllib
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from pathlib import Path


def read_google_sheet(sheet_id: str, worksheet_name: str, secrets_path: str = ".streamlit/secrets.toml") -> pd.DataFrame:
    """
    Standalone function to read a Google Sheet using credentials from secrets.toml

    Parameters:
    -----------
    sheet_id : str
        Google Sheet ID (from the URL or stored in secrets.toml under [general])
    worksheet_name : str
        Name of the worksheet/tab to read
    secrets_path : str, optional
        Path to the secrets.toml file (default: ".streamlit/secrets.toml")

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the sheet data

    Example:
    --------
    # Read from specific sheet
    df = read_google_sheet("1XsrG5_bvGtN_CyAXlo2viaUqQppd7J6iznaJn86Efi8", "GTT_DB")

    # Or use sheet_id from secrets.toml
    df = read_google_sheet(None, "GTT_DB")  # Will auto-load from secrets
    """

    # Define Google API scopes
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    # Load secrets from TOML file
    secrets_file = Path(secrets_path)
    if not secrets_file.exists():
        raise FileNotFoundError(f"Secrets file not found at: {secrets_path}")

    with open(secrets_file, "rb") as f:
        secrets = tomllib.load(f)

    # Extract GCP service account credentials
    if "gcp_service_account" not in secrets:
        raise ValueError("'gcp_service_account' section not found in secrets.toml")

    service_account_info = dict(secrets["gcp_service_account"])

    # If sheet_id is None, try to get it from secrets
    if sheet_id is None:
        if "general" in secrets and "sheet_id" in secrets["general"]:
            sheet_id = secrets["general"]["sheet_id"]
        else:
            raise ValueError("sheet_id not provided and not found in secrets.toml under [general]")

    # Create credentials and authorize gspread client
    credentials = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    client = gspread.authorize(credentials)

    # Open the spreadsheet
    spreadsheet = client.open_by_key(sheet_id)

    # Get the specific worksheet
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        available_sheets = [ws.title for ws in spreadsheet.worksheets()]
        raise ValueError(
            f"Worksheet '{worksheet_name}' not found. "
            f"Available worksheets: {', '.join(available_sheets)}"
        )

    # Get all values from the worksheet
    data = worksheet.get_all_values()

    if not data:
        return pd.DataFrame()

    # Convert to DataFrame (first row is header)
    df = pd.DataFrame(data[1:], columns=data[0])

    return df


# Convenience function that uses default sheet_id from secrets
def read_default_sheet(worksheet_name: str, secrets_path: str = ".streamlit/secrets.toml") -> pd.DataFrame:
    """
    Read from the default Google Sheet specified in secrets.toml

    Parameters:
    -----------
    worksheet_name : str
        Name of the worksheet/tab to read
    secrets_path : str, optional
        Path to the secrets.toml file (default: ".streamlit/secrets.toml")

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the sheet data

    Example:
    --------
    df = read_default_sheet("GTT_DB")
    """
    return read_google_sheet(sheet_id=None, worksheet_name=worksheet_name, secrets_path=secrets_path)


if __name__ == "__main__":
    # Example usage
    try:
        # Read using default sheet_id from secrets.toml
        df = read_default_sheet("GTT_DB")
        print(f"Successfully read {len(df)} rows from GTT_DB")
        print("\nFirst 5 rows:")
        print(df.head())

        # Or specify sheet_id explicitly
        # df = read_google_sheet("1XsrG5_bvGtN_CyAXlo2viaUqQppd7J6iznaJn86Efi8", "GTT_DB")

    except Exception as e:
        print(f"Error: {e}")
