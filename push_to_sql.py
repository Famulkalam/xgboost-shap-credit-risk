"""Load the raw loans and the model predictions into SQL Server.

Order of operations:
  1. Run setup_database.sql (creates dbo.Loans and dbo.LoanPredictions)
  2. python credit_risk_model.py   (writes predictions.csv)
  3. python push_to_sql.py         (this script, appends rows to both tables)
  4. Run create_views.sql          (creates dbo.vw_LoanRiskSummary for Power BI)

Credentials come from a .env file (copy .env.example). Never hardcode them.
"""

import os
import urllib.parse

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from data_prep import load_raw

load_dotenv()

SERVER = os.getenv("DB_SERVER", "localhost,1433")
DATABASE_NAME = os.getenv("DB_NAME", "CreditRiskDB")
USERNAME = os.getenv("DB_USERNAME", "sa")
PASSWORD = os.getenv("DB_PASSWORD")
DRIVER = os.getenv("DB_DRIVER", "{ODBC Driver 18 for SQL Server}")


def get_engine():
    if not PASSWORD:
        raise SystemExit("DB_PASSWORD is not set. Copy .env.example to .env and fill it in.")
    conn_str = (
        f"DRIVER={DRIVER};SERVER={SERVER};DATABASE={DATABASE_NAME};"
        f"UID={USERNAME};PWD={PASSWORD};Encrypt=yes;TrustServerCertificate=yes;"
    )
    return create_engine(f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}")


def push_data():
    loans = load_raw()  # includes loan_id, matching predictions.csv
    preds = pd.read_csv("predictions.csv")
    preds = preds[
        ["loan_id", "probability_default", "scorecard_pd", "scorecard_score", "risk_band", "is_high_risk"]
    ]

    engine = get_engine()
    with engine.begin() as conn:
        # Clear previous runs; the tables themselves are created by setup_database.sql
        conn.execute(text("DELETE FROM dbo.LoanPredictions"))
        conn.execute(text("DELETE FROM dbo.Loans"))

    print(f"Uploading {len(loans):,} loans to dbo.Loans ...")
    loans.to_sql("Loans", engine, schema="dbo", if_exists="append", index=False, chunksize=1000)
    print(f"Uploading {len(preds):,} predictions to dbo.LoanPredictions ...")
    preds.to_sql("LoanPredictions", engine, schema="dbo", if_exists="append", index=False, chunksize=1000)
    print("Done. Now run create_views.sql and refresh Power BI.")


if __name__ == "__main__":
    push_data()
