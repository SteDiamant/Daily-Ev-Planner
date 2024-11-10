import sqlite3
import csv

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect("energy_data.db")
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
with open(
    "/Users/steliosdiamantopoulos/Desktop/deployments/DailyEVPlanner/streamlitProject/data_original.csv",
    "r",
) as file:
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
