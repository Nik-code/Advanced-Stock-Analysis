from influx_client import get_influxdb_client
from influxdb_client import Point, WritePrecision

bucket = "stock_data"


def write_test_data():
    client = get_influxdb_client()
    write_api = client.write_api(write_options=SYNCHRONOUS)

    for value in range(5):
        point = (
            Point("test_measurement")
            .tag("tagname", "test_tag")
            .field("field_name", value)
        )
        write_api.write(bucket=bucket, org="my_org", record=point)
        print(f"Written point {value} to InfluxDB")
