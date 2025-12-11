import sqlite3
import csv
import os
from pathlib import Path

default_db_path = Path(__file__).resolve().parent / "energy_data.db"
db_path = Path(os.getenv("ENERGY_DB_PATH", default_db_path))

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Create the table with column names in double quotes
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS EnergyDemand (
        "Time" TEXT,
        "General Demand (W)" INTEGER,
        "EV Demand (W)" INTEGER,
        "Heating Demand (W)" INTEGER,
        "PV (W)" INTEGER
    )
"""
)

# Open the CSV file and read the data
default_csv_path = Path(__file__).resolve().parent / "data_original.csv"
csv_path = Path(os.getenv("ENERGY_SOURCE_CSV", default_csv_path))

with open(csv_path, "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row

    # Insert each row of data into the table, with column names in double quotes
    for row in csv_reader:
        cursor.execute(
            """
            INSERT INTO EnergyDemand ("Time", "General Demand (W)", "EV Demand (W)", "Heating Demand (W)", "PV (W)")
            VALUES (?, ?, ?, ?, ?)
        """,
            row,
        )

# Commit changes and close connection
conn.commit()
conn.close()
