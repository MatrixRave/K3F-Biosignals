import influxdb_client 
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime

bucket = 'facialReco'
org = 'k3f'
token = 'EtqvKKUjatci_cjEJphKbcS9sT3zdMMLk4RLuOQvzTwYvGi3vjb_4Iy2fm-DjBaqwGmxiBXswYBvsQIXKc1wfg=='
url = 'http://localhost:8086'

client = influxdb_client.InfluxDBClient(
	url = url,
	token = token, 
	org = org,
	timeout=30_000
)

# Setup of payload 
def create_payload(leftBlinkRatio, rightBlinkRatio, combinedBlinkRatio):

	data = {
		"measurement": "blinks",
		"tag":  "BlinkValues",
		"time": datetime.now(),
		"fields": {
			'leftBlinkRatio': leftBlinkRatio,
			'rightBlinkRatio': rightBlinkRatio, 
			'combinedBlinkRatio': combinedBlinkRatio 
		}
	}
	return data
