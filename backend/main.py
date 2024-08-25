import requests
import json

BASE_URL = "http://localhost:8000"  # Adjust this if your API is hosted elsewhere

def test_root():
    response = requests.get(f"{BASE_URL}/")
    print("Root Endpoint:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_historical_data(code="500325", days=30):
    response = requests.get(f"{BASE_URL}/api/historical/{code}?days={days}")
    print(f"Historical Data for {code}:")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Number of data points: {len(data)}")
        print(f"First data point: {json.dumps(data[0], indent=2)}")
    else:
        print(f"Error: {response.text}")
    print()

def test_login():
    response = requests.get(f"{BASE_URL}/api/login")
    print("Login Endpoint:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_callback(request_token="dummy_token"):
    response = requests.get(f"{BASE_URL}/api/callback?request_token={request_token}")
    print("Callback Endpoint:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def main():
    test_root()
    test_historical_data()
    test_login()
    test_callback()

if __name__ == "__main__":
    main()