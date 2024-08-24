import asyncio
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "BSE Stock Analysis API is running"}


def test_get_stocks():
    response = client.get("/api/stocks")
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        assert isinstance(response.json(), list)
        if len(response.json()) > 0:
            # Check if each stock has the expected fields
            for stock in response.json():
                assert "scrip_code" in stock
                assert "currentValue" in stock
                assert "predicted_price" in stock
    else:
        assert response.json() == {"detail": "No stock data available"}


if __name__ == "__main__":
    test_root()
    test_get_stocks()
    print("All tests passed!")