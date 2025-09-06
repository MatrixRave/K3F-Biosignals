import csv
import os
import time
from pprint import pprint
from enum import Enum

import neurokit2 as nk
import pandas as pd
from tqdm import tqdm

from database_setup import Database

pd.set_option('display.max_columns', None)


class Shimmer3GSRplus(Enum):
    # channels for Shimmer3 GSR+ records

    EDA = "GSR_Skin_Conductance_CAL"
    PPG = "PPG_A13_CAL"

    BAT = 'Battery_CAL'
    PRESSURE = 'Pressure_BMP280_CAL'
    TEMP = 'Temperature_BMP280_CAL'
    TIMESTAMP = 'Timestamp_FormattedUnix_CAL'

    GSR_RANGE = 'GSR_Range_CAL'
    GSR_RES = 'GSR_Skin_Resistance_CAL'

    ACCEL_LN_X = 'Accel_LN_X_CAL'
    ACCEL_LN_Y = 'Accel_LN_Y_CAL'
    ACCEL_LN_Z = 'Accel_LN_Z_CAL'
    ACCEL_WR_X = 'Accel_WR_X_CAL'
    ACCEL_WR_Y = 'Accel_WR_Y_CAL'
    ACCEL_WR_Z = 'Accel_WR_Z_CAL'

    GYRO_X = 'Gyro_X_CAL'
    GYRO_Y = 'Gyro_Y_CAL'
    GYRO_Z = 'Gyro_Z_CAL'

    MAG_X = 'Mag_X_CAL'
    MAG_Y = 'Mag_Y_CAL'
    MAG_Z = 'Mag_Z_CAL'


class Shimmer3Record:

    def __init__(self, csv_file, sampling_rate_hz=30):
        self.record = None
        self.units = None
        self.sampling_rate_hz = sampling_rate_hz
        self.parse_csv_record(csv_file)
        self.drop_leading_and_trailing_rows(criteria=Shimmer3GSRplus.EDA.value, lower_threshold=0.0)

    def as_dataframe(self):
        return self.record

    def parse_csv_record(self, csv_file):
        # find seperator and read file
        sep = self.find_delimiter(csv_file)
        df = pd.read_csv(csv_file, sep=sep, low_memory=False, header=1)

        # remove device names from collum names
        df.columns = df.columns.str.replace(r"^Shimmer_[0-9A-F]{4}_", "", regex=True)

        # get units and remove respective row afterward
        self.units = df.iloc[0].to_dict()
        df = df.iloc[1:].reset_index(drop=True)
        pprint(self.units)

        if 'Timestamp_FormattedUnix_CAL' not in df.columns:
            raise ValueError("Error: Please export data with formated UNIX timestamp in Consensys!")

        # add formated timestamp col
        ts = pd.to_datetime(df[Shimmer3GSRplus.TIMESTAMP.value], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
        ts = ts.fillna(pd.to_datetime(df[Shimmer3GSRplus.TIMESTAMP.value], format="%Y/%m/%d %H:%M:%S", errors="coerce"))
        df["timestamp"] = ts

        # set dtypes
        df = df.apply(lambda s: pd.to_numeric(s, errors="ignore") if s.dtype == object else s)

        self.record = df

    @staticmethod
    def find_delimiter(fp):
        candidates = [",", ";", "\t", "|", ":"]
        with open(fp, 'r') as fobj:
            sample = fobj.read(65535)
            dialect = csv.Sniffer().sniff(sample, delimiters=candidates)
            sep = dialect.delimiter
        return str(sep)

    def drop_leading_and_trailing_rows(self, criteria=Shimmer3GSRplus.EDA.value, lower_threshold=0.0):
        df = self.record
        self.record = df[df[criteria] >= lower_threshold].reset_index(drop=True)

    def process_electrodermal_activity(self, report_out_dir=None):
        os.makedirs(report_out_dir, exist_ok=True)
        html_report_fp = os.path.join(report_out_dir, "report_shimmer3_eda.html")
        eda_signal = self.record[Shimmer3GSRplus.EDA.value]
        signals, info = nk.eda_process(eda_signal,
                                       sampling_rate=self.sampling_rate_hz,
                                       report=html_report_fp,
                                       method="neurokit")

        fig = nk.eda_plot(signals, info, static=False)
        fig.show()

        # add to dataframe
        common = self.record.columns.intersection(signals.columns)
        self.record = self.record.drop(columns=common).join(signals, how="outer")
        return signals

    def process_photoplethysmogram(self, report_out_dir=None):
        os.makedirs(report_out_dir, exist_ok=True)
        html_report_fp = os.path.join(report_out_dir, "report_shimmer3_ppg.html")
        ppg_signal = self.record[Shimmer3GSRplus.PPG.value]
        signals, info = nk.ppg_process(ppg_signal,
                                       sampling_rate=self.sampling_rate_hz,
                                       report=html_report_fp,
                                       method="elgendi",
                                       method_quality="templatematch")

        fig = nk.ppg_plot(signals, info, static=False)
        fig.show()

        # add to dataframe
        common = self.record.columns.intersection(signals.columns)
        self.record = self.record.drop(columns=common).join(signals, how="outer")
        return signals


def write_shimmer_record_to_database(record, measurement_name):
    record_df = record.as_dataframe()
    col_names = record_df.columns.to_list()
    db = Database()

    print("Writing Shimmer records to database...")
    print(col_names)
    for idx, row in tqdm(record_df.iterrows()):
        fields = {
            sig: row[sig] for sig in col_names if sig not in ['timestamp',]
        }
        tags = {
            "bioSignal": "eda_ppg",
            "device": "Shimmer3 GSR+"
        }
        ts = pd.to_datetime(row['timestamp'])
        if ts.tzinfo is None:
            ts = ts.tz_localize("Europe/Berlin")
        timestamp_utc = ts.tz_convert("UTC")

        db.write_record(measurement_name,
                        tags,
                        fields,
                        timestamp_utc)

    time.sleep(1) # time for db write buffer to flush


if __name__ == '__main__':
    data_fp = "../../sample_data/trial_1/2025-08-25_15.50.55_K3FTrial_SD_Session1_localtimestamp/K3FTrial_Session1_Shimmer_A66F_Calibrated_SD.csv"
    report_dir = "../../reports"

    # read recorded Shimmer3 GSR+ data (CSV export from Consensys with local timestamp)
    record = Shimmer3Record(data_fp)

    # analyze biosignals
    record.process_electrodermal_activity(report_out_dir=report_dir)
    record.process_photoplethysmogram(report_out_dir=report_dir)
    print(record.as_dataframe().head())

    # write into InfluxDB
    write_shimmer_record_to_database(record, measurement_name='shimmer3gsrplus')

