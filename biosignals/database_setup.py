import influxdb_client


class Database:
    BUCKET = 'shimmer'
    ORG = 'k3f'
    TOKEN = 'EtqvKKUjatci_cjEJphKbcS9sT3zdMMLk4RLuOQvzTwYvGi3vjb_4Iy2fm-DjBaqwGmxiBXswYBvsQIXKc1wfg=='
    URL = 'http://localhost:8086'

    def __init__(self, bucket=None, org=None, token=None, url=None):
        self.bucket = bucket if bucket else Database.BUCKET
        self.org = org if org else Database.ORG
        self.token = token if token else Database.TOKEN
        self.url = url if url else Database.URL
        self.timeout = 30_000

        self.client = influxdb_client.InfluxDBClient(
            url=self.url,
            token=self.token,
            org=self.org,
            timeout=self.timeout
        )
        self.write_api = self.client.write_api()

    def write_record(self, measurement_name, tags, fields, timestamp_utc=None):
        data = influxdb_client.Point(measurement_name)
        for name, value in tags.items():
            data.tag(name, value)
        for name, value in fields.items():
            data.field(name, value)

        # explicit timestamp
        if timestamp_utc is not None:
            data.time(timestamp_utc.isoformat())

        self.write_api.write(bucket=self.bucket,
                             org=self.org,
                             record=data)

