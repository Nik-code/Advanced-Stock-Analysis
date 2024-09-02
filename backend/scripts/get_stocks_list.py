from kiteconnect import KiteConnect
import pandas as pd
import os
import dotenv

dotenv.load_dotenv()

# Initialize Kite Connect
api_key = os.getenv("ZERODHA_API_KEY")
access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Fetch all instruments
instruments = kite.instruments()

# Convert to DataFrame
df = pd.DataFrame(instruments)

# Filter for Indian stocks (NSE and BSE)
indian_stocks = df[(df['exchange'].isin(['NSE', 'BSE'])) & (df['instrument_type'] == 'EQ')]

# Select relevant columns
columns_to_keep = ['tradingsymbol', 'name', 'exchange', 'instrument_token', 'exchange_token']
indian_stocks = indian_stocks[columns_to_keep]

# Save to CSV
csv_filename = 'indian_stocks.csv'
indian_stocks.to_csv(csv_filename, index=False)

print(f"CSV file '{csv_filename}' has been created with {len(indian_stocks)} Indian stocks.")