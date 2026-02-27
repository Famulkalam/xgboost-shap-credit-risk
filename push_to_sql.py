import os
import pandas as pd
import pyodbc 
from sqlalchemy import create_engine
import urllib
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION (via .env file) ---
SERVER = os.getenv('DB_SERVER', 'localhost,1433')
DATABASE_NAME = os.getenv('DB_NAME', 'CreditRiskDB')
USERNAME = os.getenv('DB_USERNAME', 'sa')
PASSWORD = os.getenv('DB_PASSWORD')  # Set in .env file — NEVER hardcode
DRIVER = os.getenv('DB_DRIVER', '{ODBC Driver 18 for SQL Server}')

def push_data():
    try:
        # 1. Load Data
        print("Reading CSV files...")
        df_loans = pd.read_csv('credit_risk_dataset.csv')
        df_preds = pd.read_csv('predictions.csv')

        # 2. Connection for Database Creation (Connect to 'master')
        base_conn_str = f"DRIVER={DRIVER};SERVER={SERVER};UID={USERNAME};PWD={PASSWORD};Encrypt=yes;TrustServerCertificate=yes;"
        
        # Create Database if it doesn't exist
        print(f"Ensuring database '{DATABASE_NAME}' exists...")
        conn = pyodbc.connect(base_conn_str, autocommit=True)
        cursor = conn.cursor()
        cursor.execute(f"IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = '{DATABASE_NAME}') CREATE DATABASE {DATABASE_NAME}")
        conn.close()

        # 3. Connection for Data Upload (SQLAlchemy)
        upload_params = urllib.parse.quote_plus(f"{base_conn_str}DATABASE={DATABASE_NAME};")
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={upload_params}")

        print(f"Connecting to SQL Server: {SERVER}...")

        # 4. Upload Raw Loans
        print("Uploading raw loans to dbo.Loans...")
        df_loans.to_sql('Loans', con=engine, if_exists='replace', index=False)
        
        # 5. Upload Predictions
        print("Uploading predictions to dbo.LoanPredictions...")
        df_preds.to_sql('LoanPredictions', con=engine, if_exists='replace', index=False)

        print("\nSUCCESS: All data pushed to MS SQL Server!")
        print(f"Database: {DATABASE_NAME} | Tables: Loans, LoanPredictions")
        print("You can now refresh your Power BI dashboard.")

    except Exception as e:
        print("\nERROR pushing to SQL:")
        print(f"Details: {e}")
        print("\nVERIFY: Is your Docker container running? Is your password correct in push_to_sql.py?")

if __name__ == "__main__":
    push_data()
