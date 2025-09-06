import os
import re
import pandas as pd
from datetime import datetime, timedelta
from influxdb_client import Point, WritePrecision
from database import Database
from tqdm import tqdm
import time

def extract_start_time_from_filename(filename: str) -> datetime:
    match = re.match(r"(\d{8})-(\d{6})", filename)
    if not match:
        match = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
    date_str, time_str = match.groups()
    return datetime.strptime(date_str + time_str, "%d%m%Y%H%M%S")

def import_csv_to_influx(file_path: str):
    database = Database()
    filename = os.path.basename(file_path)
    start_time = extract_start_time_from_filename(filename)

    # Read CSV, skipping first 20 lines
    df = pd.read_csv(file_path, skiprows=20)

    # First column is duration in milliseconds
    duration_col = df.columns[0]
    df["timestamp"] = [start_time + timedelta(milliseconds=val) for val in df[duration_col]]

    buffer = []
    for _, row in tqdm(df.iterrows()):
        point = (
            Point("csv_import")  # measurement name
            .time(row["timestamp"], WritePrecision.NS)
        )
        for col in df.columns:
            if col not in [duration_col, "timestamp"]:
                value = row[col]
                if pd.notna(value):
                    try:
                        point.field(col, float(value))
                    except ValueError:
                        point.field(col, str(value))
        buffer.append(point)

        # Write batch when buffer is full
        if len(buffer) >= database.BATCH_SIZE:
            database.write_batch(buffer)
            buffer = []

    # Write any remaining points
    if buffer:
        database.write_batch(buffer)

    time.sleep(1)


def import_multiple(files):
    for f in files:
        print("Importing telemtry from", f)
        import_csv_to_influx(f)

if __name__ == "__main__":
    # Import all CSV files in current folder
    folder = "../sample_data/telemetry"
    csv_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
    import_multiple(csv_files)
