import os
import pandas as pd
import logging
from influxdb_client import Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS  # Ensure this is imported
from influx_client import get_influxdb_client
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Script Purpose:
# This script processes stock CSV files, calculates technical indicators (including RSI),
# and ingests the data into an InfluxDB instance. It handles multiple CSV files, processes
# them in batches, and optionally runs the processing in parallel for faster execution.

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

bucket = "stock_data"
client = get_influxdb_client()


def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI) with a default 14-day window."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def read_and_process_csv(file_path):
    try:
        # Log start of file processing
        logger.info(f"Starting processing for {file_path}")

        # Read the CSV
        df = pd.read_csv(file_path)

        # Rename columns to match expected names
        df.rename(columns={
            'Open Price': 'Open',
            'High Price': 'High',
            'Low Price': 'Low',
            'Close Price': 'Close',
            'No.of Shares': 'Volume'
        }, inplace=True)

        # Log the columns found in the CSV after renaming
        logger.info(f"Columns found in {file_path}: {list(df.columns)}")

        # Validate essential columns
        required_columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"Missing essential columns in {file_path}")

        # Convert columns to numeric values (handling commas)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
        df['High'] = pd.to_numeric(df['High'], errors='coerce')
        df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

        # Convert 'Date' to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Drop rows with missing or malformed data
        df.dropna(subset=['Date', 'Close', 'Open', 'High', 'Low', 'Volume'], inplace=True)

        # Calculate indicators
        df['RSI'] = calculate_rsi(df['Close'])
        df['Moving_Average_50'] = df['Close'].rolling(window=50).mean()
        df['Moving_Average_200'] = df['Close'].rolling(window=200).mean()
        df['Volatility'] = df['Close'].rolling(window=20).std()

        return df

    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {file_path} - {fnf_error}")
    except ValueError as val_error:
        logger.error(f"Value error in {file_path}: {val_error}")
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None


def write_stock_data_to_influxdb(df, symbol, batch_size=100):
    try:
        # Batch processing
        write_api = client.write_api(write_options=SYNCHRONOUS)
        points = []

        for i, row in df.iterrows():
            point = (
                Point("stock_data")
                .tag("symbol", symbol)
                .field("close", row['close'])
                .field("open", row['open'])
                .field("high", row['high'])
                .field("low", row['low'])
                .field("volume", row['volume'])
                .field("rsi", row['RSI'])
                .field("sma_20", row['SMA_20'])
                .field("ema_50", row['EMA_50'])
                .field("macd", row['MACD'])
                .field("macd_signal", row['MACD_Signal'])
                .field("macd_hist", row['MACD_Hist'])
                .field("bollinger_upper", row['Bollinger_Upper'])
                .field("bollinger_middle", row['Bollinger_Middle'])
                .field("bollinger_lower", row['Bollinger_Lower'])
                .field("atr", row['ATR'])
                .time(row['date'], WritePrecision.NS)
            )
            points.append(point)

            # Write in batches of 'batch_size'
            if len(points) >= batch_size:
                write_api.write(bucket=bucket, org="my_org", record=points)
                points.clear()  # Clear the list after writing the batch

        # Write any remaining points
        if points:
            write_api.write(bucket=bucket, org="my_org", record=points)

        logger.info(f"Data for {symbol} written to InfluxDB.")
    except Exception as e:
        logger.error(f"Error writing data for {symbol} to InfluxDB: {str(e)}")


def process_all_csv_files():
    data_folder = os.path.join(os.path.dirname(__file__), '../../data/')

    # Use a progress bar
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    with tqdm(total=len(csv_files)) as progress_bar:
        for file_name in csv_files:
            file_path = os.path.join(data_folder, file_name)
            symbol = file_name.split('.')[0]
            df = read_and_process_csv(file_path)
            if df is not None:
                write_stock_data_to_influxdb(df, symbol)
            progress_bar.update(1)


def process_files_in_parallel():
    data_folder = os.path.join(os.path.dirname(__file__), '../../data/')
    csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]

    # Limit the number of concurrent processes
    with ProcessPoolExecutor(max_workers=8) as executor:  # Adjust based on your CPU and system load
        list(tqdm(executor.map(process_csv_file, csv_files), total=len(csv_files)))


def process_csv_file(file_path):
    symbol = os.path.basename(file_path).split('.')[0]
    df = read_and_process_csv(file_path)
    if df is not None:
        write_stock_data_to_influxdb(df, symbol)


if __name__ == "__main__":
    # Uncomment either of these based on your preference
    # process_all_csv_files()  # Single-threaded
    process_files_in_parallel()  # Parallel processing
