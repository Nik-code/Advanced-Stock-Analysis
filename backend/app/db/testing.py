import os
import time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS  # Ensure this is imported

# Use your token and configuration
token = os.environ.get("INFLUXDB_TOKEN")  # Ensure this environment variable is set
org = "my_org"  # Replace with your actual organization name
url = "http://localhost:8086"  # InfluxDB URL, likely the default one

# Create a client instance
client = InfluxDBClient(url=url, token=token, org=org)

# Specify your bucket (replace with your actual bucket name)
bucket = "stock_data"

# Initialize the write API
write_api = client.write_api(write_options=SYNCHRONOUS)

# Write test data points
for value in range(5):
    point = (
        Point("test_measurement")  # Measurement name, adjust as needed
        .tag("tagname", "test_tag")  # Add a tag (metadata)
        .field("field_name", value)  # Add a field (the actual data)
    )
    write_api.write(bucket=bucket, org=org, record=point)  # Write the point to InfluxDB
    print(f"Written point {value} to InfluxDB")
    time.sleep(1)  # Separate points by
