import os
from influxdb_client import InfluxDBClient

token = os.environ.get("INFLUXDB_TOKEN")
org = "my_org"
url = "http://localhost:8086"

def get_influxdb_client():
    return InfluxDBClient(url=url, token=token, org=org)
