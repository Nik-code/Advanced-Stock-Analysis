from influxdb_client import InfluxDBClient
import os

def get_influxdb_client():
    token = os.environ.get("INFLUXDB_TOKEN")
    org = "my_org"
    url = "http://localhost:8086"

    # Set longer timeouts for handling larger loads in parallel processing
    client = InfluxDBClient(url=url, token=token, org=org, timeout=60000)  # Set a 60s timeout

    return client
