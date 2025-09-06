import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

class Database: 

	BUCKET = 'telemetry'
	ORG = 'k3f'
	TOKEN = 'EtqvKKUjatci_cjEJphKbcS9sT3zdMMLk4RLuOQvzTwYvGi3vjb_4Iy2fm-DjBaqwGmxiBXswYBvsQIXKc1wfg=='
	URL = 'http://localhost:8086'
	BATCH_SIZE = 500 

	def __init__(self, bucket=None, org=None, token=None, url=None):
		self.bucket = bucket if bucket else Database.BUCKET
		self.org = org if org else Database.ORG
		self.token = token if token else Database.TOKEN
		self.url = url if url else Database.URL
		self.timeout = 1_000

		self.client = influxdb_client.InfluxDBClient(
			url = self.url,
			token = self.token,
			org = self.org,
			timeout=self.timeout
		)
		self.write_api = self.client.write_api()
	
	def write_batch(self, buffer):
		self.write_api.write(bucket=self.bucket,
							 org=self.org,
							 record=buffer)
