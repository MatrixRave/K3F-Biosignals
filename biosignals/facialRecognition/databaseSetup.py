import influxdb_client 
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime

bucket = 'facialReco'
org = 'k3f'
token = '3TXrWSppn__jhpiGxxxno4-fNOC2GlqYW8DGSgzheMjivEc9fEh-uVfOeg-fson3frPcp8lfPkS2P73MBERB1g=='
url = 'http://localhost:8086'

client = influxdb_client.InfluxDBClient(
	url = url,
	token = token, 
	org = org
)

# Setup of payload 
def create_payload(leftBlinkRatio, rightBlinkRatio, combinedBlinkRatio):

	data = {
		"measurement": "blinks",
		"tags": {
			"ticker": "BlinkValues" 
		}, 
		"time": datetime.now(),
		"fields": {
			'leftBlinkRatio': leftBlinkRatio,
			'rightBlinkRatio': rightBlinkRatio, 
			'combinedBlinkRatio': combinedBlinkRatio 
		}
	}
	return data
