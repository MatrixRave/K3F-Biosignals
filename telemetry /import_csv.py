import os
import re
import pandas as pd
from datetime import datetime, timedelta
from influxdb_client import Point, WritePrecision
from database import Database
from tqdm import tqdm
import time
pd.set_option('display.max_columns', None)


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

    # Read CSV, skipping first 18 lines to have a leading row with headers
    df = pd.read_csv(file_path, skiprows=18, low_memory=False)

    #remove row containing units
    df = df.iloc[1:].reset_index(drop=True)

    # set dtypes
    df = df.apply(lambda s: pd.to_numeric(s) if s.dtype == object else s)

    print(df.head())
    # First column is duration in milliseconds
    duration_col = df.columns[0]

    df["timestamp"] = [start_time + timedelta(seconds=float(val)) for val in df[duration_col]]

    measurement_name = "csv_import"
    for _, row in tqdm(df.iterrows(), total=len(df)):
        tags = {
            "device": "simulator"
        }
        ts = pd.to_datetime(row['timestamp'])
        if ts.tzinfo is None:
            ts = ts.tz_localize("Europe/Berlin")
        timestamp_utc = ts.tz_convert("UTC")

        fields = {}
        for col in df.columns:
            if col not in [duration_col, "timestamp"]:
                value = row[col]
                if pd.notna(value):
                    try:
                        fields.update({str(col): float(value)})
                    except ValueError:
                        fields.update({str(col): str(value)})

        database.write_record(measurement_name,
                              tags,
                              fields,
                              timestamp_utc)

    time.sleep(1)


def import_multiple(files):
    for f in files:
        print("Importing telemetry from:", f)
        import_csv_to_influx(f)

if __name__ == "__main__":
    # Import all CSV files in current folder
    folder = "../sample_data/telemetry"
    csv_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
    import_multiple(csv_files)
