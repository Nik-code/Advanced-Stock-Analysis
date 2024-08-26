import requests
import json

BASE_URL = "http://localhost:8000"

def test_root():
    response = requests.get(f"{BASE_URL}/")
    print("Root Endpoint:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_historical_data(symbol="RELIANCE", days=30):
    response = requests.get(f"{BASE_URL}/api/stocks/{symbol}/data?days={days}")
    print(f"Historical Data for {symbol}:")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Number of data points: {len(data['data'])}")
        if data['data']:
            print(f"First data point: {json.dumps(data['data'][0], indent=2)}")
        else:
            print("No data points available")
    else:
        print(f"Error: {response.text}")
    print()

def test_technical_indicators(symbol="RELIANCE", days=30):
    response = requests.get(f"{BASE_URL}/api/stocks/{symbol}/indicators?days={days}")
    print(f"Technical Indicators for {symbol}:")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Indicators: {json.dumps(data, indent=2)}")
    else:
        print(f"Error: {response.text}")
    print()

def test_realtime_data(symbol="RELIANCE"):
    response = requests.get(f"{BASE_URL}/api/stocks/{symbol}/realtime")
    print(f"Realtime Data for {symbol}:")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Realtime data: {json.dumps(data, indent=2)}")
    else:
        print(f"Error: {response.text}")
    print()

def test_market_overview(limit=5):
    response = requests.get(f"{BASE_URL}/api/market/overview?limit={limit}")
    print(f"Market Overview (Top {limit} stocks):")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Overview: {json.dumps(data, indent=2)}")
    else:
        print(f"Error: {response.text}")
    print()

def test_compare_stocks(symbols="RELIANCE,TCS", days=30):
    response = requests.get(f"{BASE_URL}/api/stocks/compare?symbols={symbols}&days={days}")
    print(f"Compare Stocks ({symbols}):")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Comparison data: {json.dumps(data, indent=2)}")
    else:
        print(f"Error: {response.text}")
    print()

if __name__ == "__main__":
    test_root()
    test_historical_data()
    test_technical_indicators()
    test_realtime_data()
    test_market_overview()
    test_compare_stocks()