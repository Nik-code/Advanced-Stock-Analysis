from influxdb_client import InfluxDBClient
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# InfluxDB connection details
token = os.environ.get("INFLUXDB_TOKEN")
org = "my_org"
bucket = "stock_data"
url = "http://localhost:8086"

def query_influxdb(scrip_code):
    client = InfluxDBClient(url=url, token=token, org=org)
    query_api = client.query_api()

    query = f'''
    from(bucket: "{bucket}")
      |> range(start: -1y)
      |> filter(fn: (r) => r._measurement == "stock_data" and r.symbol == "{scrip_code}")
      |> limit(n: 10)
    '''
    
    logger.info(f"Executing query: {query}")
    
    try:
        result = query_api.query(org=org, query=query)
        logger.info(f"Query result for {scrip_code}:")
        for table in result:
            for record in table.records:
                logger.info(f"Time: {record.get_time()}, Fields: {record.values}")
    except Exception as e:
        logger.error(f"Error querying InfluxDB: {e}")

if __name__ == "__main__":
    scrip_code = "RELIANCE"  # Change this to the stock code you want to query
    query_influxdb(scrip_code)