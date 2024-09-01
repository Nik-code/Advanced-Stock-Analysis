from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from app.services.data_collection import fetch_historical_data
from app.services.technical_indicators import calculate_sma, calculate_ema, calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_atr
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv, find_dotenv, set_key
import logging
from app.services.zerodha_service import ZerodhaService
from app.api import stocks
import os
import pandas as pd
import numpy as np
from typing import List
from app.models.lstm_model import LSTMStockPredictor
import joblib

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()  # This line is crucial

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scheduler = AsyncIOScheduler()
zerodha_service = ZerodhaService()

app.include_router(stocks.router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "BSE Stock Analysis API is running"}


@app.get("/api/quote")
async def get_quote(instruments: str):
    try:
        quote_data = zerodha_service.get_quote(instruments)
        if quote_data is None:
            raise HTTPException(status_code=404, detail="Quote not found")
        return quote_data
    except Exception as e:
        logger.error(f"Error fetching quote for {instruments}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching quote")


@app.get("/api/historical/{code}")
async def get_historical_data(code: str, timeFrame: str = '1year'):
    try:
        data = await fetch_historical_data(code, timeFrame)
        if data is None:
            raise HTTPException(status_code=404, detail=f"No data found for stock code {code}")
        return data.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error fetching historical data for {code}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/login")
async def login():
    login_url = zerodha_service.get_login_url()
    return {"login_url": login_url}


@app.get("/api/callback")
async def callback(request: Request):
    params = dict(request.query_params)
    request_token = params.get("request_token")
    if not request_token:
        raise HTTPException(status_code=400, detail="No request token provided")
    try:
        access_token = zerodha_service.generate_session(request_token)
        zerodha_service.set_access_token(access_token)
        # Save the access token to .env file
        dotenv_file = find_dotenv()
        set_key(dotenv_file, "ZERODHA_ACCESS_TOKEN", access_token)
        return {"access_token": access_token}
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating session")


@app.get("/api/stocks/{symbol}/indicators")
async def get_technical_indicators(symbol: str, timeFrame: str = '1year'):
    try:
        historical_data = await fetch_historical_data(symbol, timeFrame)
        if historical_data is None:
            raise HTTPException(status_code=404, detail=f"No data found for stock symbol {symbol}")

        logger.info(f"Columns in the dataframe: {historical_data.columns}")
        logger.info(f"Number of data points: {len(historical_data)}")

        # Calculate technical indicators
        sma_20 = calculate_sma(historical_data['close'], 20)
        ema_50 = calculate_ema(historical_data['close'], 50)
        rsi_14 = calculate_rsi(historical_data['close'], 14)
        macd, signal, _ = calculate_macd(historical_data['close'])
        upper, middle, lower = calculate_bollinger_bands(historical_data['close'])
        atr = calculate_atr(historical_data['high'], historical_data['low'], historical_data['close'], 14)

        def safe_float_list(arr):
            return [float(x) if not np.isnan(x) and not np.isinf(x) else None for x in arr]

        return {
            "sma_20": safe_float_list(sma_20),
            "ema_50": safe_float_list(ema_50),
            "rsi_14": safe_float_list(rsi_14),
            "macd": safe_float_list(macd),
            "macd_signal": safe_float_list(signal),
            "bollinger_upper": safe_float_list(upper),
            "bollinger_middle": safe_float_list(middle),
            "bollinger_lower": safe_float_list(lower),
            "atr": safe_float_list(atr),
            "dates": historical_data['date'].tolist(),
            "close_prices": safe_float_list(historical_data['close'])
        }
    except Exception as e:
        logger.error(f"Error calculating technical indicators for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/{symbol}/realtime")
async def get_realtime_data(symbol: str):
    try:
        instrument_token = zerodha_service.get_instrument_token("BSE", symbol)
        if not instrument_token:
            raise HTTPException(status_code=404, detail=f"No instrument token found for stock symbol {symbol}")
        realtime_data = zerodha_service.get_quote([instrument_token])
        if realtime_data is None or str(instrument_token) not in realtime_data:
            raise HTTPException(status_code=404, detail=f"No real-time data found for stock symbol {symbol}")
        return realtime_data[str(instrument_token)]
    except Exception as e:
        logger.error(f"Error fetching real-time data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/overview")
async def get_market_overview(limit: int = 10):
    try:
        # This is a placeholder. You should replace it with actual top stocks from your data
        top_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "HDFC", "SBIN", "BHARTIARTL", "ITC"][:limit]
        
        overview_data = []
        for symbol in top_stocks:
            quote = zerodha_service.get_quote(f"BSE:{symbol}")
            if quote and f"BSE:{symbol}" in quote:
                stock_data = quote[f"BSE:{symbol}"]
                overview_data.append({
                    "symbol": symbol,
                    "last_price": stock_data.get("last_price"),
                    "change": stock_data.get("change"),
                    "change_percent": stock_data.get("change_percent")
                })
        
        return overview_data
    except Exception as e:
        logger.error(f"Error fetching market overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/compare")
async def compare_stocks(symbols: str, days: int = 365):
    try:
        symbol_list = symbols.split(",")
        if len(symbol_list) < 2:
            raise HTTPException(status_code=400, detail="Please provide at least two stock symbols for comparison")

        comparison_data = {}
        for symbol in symbol_list:
            data = await fetch_historical_data(symbol, days)
            if data is None:
                raise HTTPException(status_code=404, detail=f"No data found for stock symbol {symbol}")

            logger.info(f"Columns in the dataframe for {symbol}: {data.columns}")

            # Check if 'close' column exists (case-insensitive)
            close_column = next((col for col in data.columns if col.lower() == 'close'), None)
            if not close_column:
                raise ValueError(f"No 'close' column found for {symbol}. Available columns: {', '.join(data.columns)}")

            close_prices = data[close_column]
            comparison_data[symbol] = {
                "prices": close_prices.tolist(),
                "return": ((close_prices.iloc[-1] / close_prices.iloc[0]) - 1) * 100,
                "volatility": close_prices.pct_change().std() * (252 ** 0.5) * 100  # Annualized volatility
            }

        return comparison_data
    except Exception as e:
        logger.error(f"Error comparing stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/{stock_code}")
async def predict_stock(stock_code: str, data: List[float]):
    try:
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        model_path = os.path.join(model_dir, f'{stock_code}_lstm_model.h5')
        scaler_path = os.path.join(model_dir, f'{stock_code}_scaler.pkl')
        
        predictor = LSTMStockPredictor(input_shape=(60, 1))
        predictor.load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        data = np.array(data).reshape(-1, 1)
        scaled_data = scaler.transform(data)
        X = np.array([scaled_data[-60:]])
        prediction = predictor.predict(X)
        prediction = scaler.inverse_transform(prediction)
        return {"prediction": prediction[0][0]}
    except Exception as e:
        logger.error(f"Error making prediction for {stock_code}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction for {stock_code}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
