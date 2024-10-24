import os

import joblib
import logging
import numpy as np

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import find_dotenv, load_dotenv, set_key
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from app.api import stocks
from app.services.data_collection import fetch_historical_data, fetch_news_data
from app.services.llm_integration import GPT4Processor
from app.services.technical_indicators import calculate_sma, calculate_ema, calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_atr
from app.services.zerodha_service import ZerodhaService

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
llm_processor = GPT4Processor()

app.include_router(stocks.router, prefix="/api/stocks")


@app.get("/")
async def root():
    return {"message": "BSE Stock Analysis API is running"}


@app.get("/api/quote")
async def get_quote(instruments: str):
    """
    Fetch the latest market quote for the specified instruments.

    This endpoint retrieves the latest trading information for the given stock instruments.

    Parameters:
    - instruments (str): A comma-separated list of instrument codes (e.g., "BSE:RELIANCE").

    Example of a request:
    http://localhost:8000/api/quote?instruments=BSE:RELIANCE

    Returns:
    - dict: A dictionary containing the latest quote data for the specified instruments.
            The structure of the response is as follows:
            {
                "BSE:RELIANCE": {
                    "instrument_token": 128083204,
                    "timestamp": "2024-09-06T16:01:32",
                    "last_trade_time": "2024-09-06T15:59:57",
                    "last_price": 2929.85,
                    "last_quantity": 0,
                    "buy_quantity": 0,
                    "sell_quantity": 0,
                    "volume": 0,
                    "average_price": 0,
                    "oi": 0,
                    "oi_day_high": 0,
                    "oi_day_low": 0,
                    "net_change": 0,
                    "lower_circuit_limit": 2636.9,
                    "upper_circuit_limit": 3222.8,
                    "ohlc": {
                        "open": 2989.9,
                        "high": 2996.2,
                        "low": 2922.75,
                        "close": 2987.15
                    },
                    "depth": {
                        "buy": [
                            {"price": 0, "quantity": 0, "orders": 0},
                            {"price": 0, "quantity": 0, "orders": 0},
                            {"price": 0, "quantity": 0, "orders": 0},
                            {"price": 0, "quantity": 0, "orders": 0},
                            {"price": 0, "quantity": 0, "orders": 0}
                        ],
                        "sell": [
                            {"price": 0, "quantity": 0, "orders": 0},
                            {"price": 0, "quantity": 0, "orders": 0},
                            {"price": 0, "quantity": 0, "orders": 0},
                            {"price": 0, "quantity": 0, "orders": 0},
                            {"price": 0, "quantity": 0, "orders": 0}
                        ]
                    }
                }
            }

    Raises:
    - HTTPException: If the quote is not found or if an internal error occurs.
    """
    try:
        quote_data = zerodha_service.get_quote(instruments)
        if quote_data is None:
            raise HTTPException(status_code=404, detail="Quote not found")
        return quote_data
    except Exception as e:
        logger.error(f"Error fetching quote for {instruments}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching quote")


@app.get("/api/predict/{stock_code}")
async def predict_stock(stock_code: str):
    try:
        logger.info(f"Received prediction request for {stock_code}")
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'trained_models', stock_code)
        lstm_model_path = os.path.join(model_dir, 'LSTM_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        arima_model_path = os.path.join(model_dir, 'ARIMA_model.pkl')

        if not os.path.exists(lstm_model_path) or not os.path.exists(scaler_path) or not os.path.exists(arima_model_path):
            logger.error(f"Models or scaler not found for {stock_code}")
            raise HTTPException(status_code=404, detail=f"Models not found for {stock_code}")

        lstm_predictor = LSTMStockPredictor(input_shape=(60, 1))
        lstm_predictor.load_model(lstm_model_path)
        scaler = joblib.load(scaler_path)
        arima_predictor = joblib.load(arima_model_path)

        # Fetch historical data
        historical_data = await fetch_historical_data(stock_code, '1year')
        if historical_data is None:
            raise HTTPException(status_code=404, detail=f"No historical data found for {stock_code}")

        close_prices = historical_data['close'].values.reshape(-1, 1)
        scaled_data = scaler.transform(close_prices)

        # Generate past predictions for LSTM
        lstm_past_predictions = []
        for i in range(60, len(scaled_data)):
            X = scaled_data[i-60:i].reshape(1, 60, 1)
            prediction = lstm_predictor.predict(X)
            lstm_past_predictions.append(prediction[0][0])

        lstm_past_predictions = scaler.inverse_transform(np.array(lstm_past_predictions).reshape(-1, 1))

        # Generate future predictions for LSTM
        lstm_future_predictions = []
        last_sequence = scaled_data[-60:].reshape(1, 60, 1)
        for _ in range(7):  # Predict next 7 days
            prediction = lstm_predictor.predict(last_sequence)
            lstm_future_predictions.append(prediction[0][0])
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = prediction[0][0]

        lstm_future_predictions = scaler.inverse_transform(np.array(lstm_future_predictions).reshape(-1, 1))

        # Generate predictions for ARIMA
        arima_past_predictions = arima_predictor.predict(len(close_prices) - 60)
        arima_future_predictions = arima_predictor.predict(7)

        logger.info(f"Successfully generated predictions for {stock_code}")

        return {
            "lstm_past_predictions": lstm_past_predictions.flatten().tolist(),
            "lstm_future_predictions": lstm_future_predictions.flatten().tolist(),
            "arima_past_predictions": arima_past_predictions.tolist(),
            "arima_future_predictions": arima_future_predictions.tolist()
        }
    except Exception as e:
        logger.error(f"Error making prediction for {stock_code}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction for {stock_code}: {str(e)}")


@app.get("/api/analysis/{stock_code}")
async def get_stock_analysis(stock_code: str):
    try:
        historical_data = await fetch_historical_data(stock_code, '1year')
        news_data = await fetch_news_data(stock_code)
        lstm_prediction = await get_lstm_prediction(stock_code, historical_data)
        arima_prediction = await get_arima_prediction(stock_code, historical_data)

        analysis = await llm_processor.analyze_stock(
            stock_code,
            historical_data,
            news_data,
            lstm_prediction,
            arima_prediction
        )

        return analysis
    except Exception as e:
        logger.error(f"Error generating analysis for {stock_code}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/top_stocks")
async def get_top_stocks():
    try:
        # Implement logic to analyze all stocks and return top picks
        top_stocks = await analyze_all_stocks()
        return {"top_stocks": top_stocks}
    except Exception as e:
        logger.error(f"Error getting top stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Scheduled task to run predictions and make trade decisions
@app.on_event("startup")
async def run_predictions_and_trades():
    try:
        # Implement logic to run predictions and make trade decisions
        await run_predictions_and_make_trades()
    except Exception as e:
        logger.error(f"Error in scheduled task: {str(e)}")


@app.get("/api/login")
async def login():
    """
    Endpoint to initiate the login process for the Zerodha trading platform.

    This endpoint returns the login URL that the user needs to visit in order to authenticate
    and generate a session token. The user will be redirected to the Zerodha login page,
    where they can enter their credentials.

    Returns:
    - dict: A dictionary containing the login URL.

    Example of a successful response:
    {
        "login_url": "https://zerodha.com/login"
    }
    """
    login_url = zerodha_service.get_login_url()
    return {"login_url": login_url}


@app.get("/api/callback")
async def callback(request: Request):
    """
    Endpoint to handle the callback from the Zerodha login process.

    This endpoint is called after the user has authenticated with Zerodha and is redirected back
    to the application. It retrieves the request token from the query parameters, generates an
    access token, and saves it for future use.

    Parameters:
    - request (Request): The incoming request containing the query parameters.

    Returns:
    - dict: A dictionary containing the access token.

    Raises:
    - HTTPException: If no request token is provided (400 Bad Request).
    - HTTPException: If there is an error generating the session (500 Internal Server Error).

    Example of a successful response:
    {
        "access_token": "your_access_token_here"
    }
    """
    # Log the full request details to check for the request_token
    logger.info(f"Request params: {request.query_params}")

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


class PredictionRequest(BaseModel):
    data: List[float]


class BacktestRequest(BaseModel):
    stock_code: str
    days: str = '1year'
    model: str = 'lstm'

@app.post("/api/backtest")
async def backtest_stock(request: BacktestRequest):
    """
    Backtest the specified model (LSTM or ARIMA) for a given stock over a specified time period.

    Parameters:
    - stock_code (str): The stock symbol for which backtesting is requested (e.g., "RELIANCE").
    - days (str): The time frame for historical data. Default is '1year'.
                  It can be one of the following values: '1month', '3months', '1year', '5years'
    - model (str): The model to use for backtesting. Default is 'lstm'.
                   It can be either 'lstm' or 'arima'.

    Returns:
    - dict: A dictionary containing the backtesting results.

    Raises:
    - HTTPException: If no historical data is found or if an error occurs during backtesting.
    """
    try:
        logger.info(f"Received backtesting request for {request.stock_code} using {request.model.upper()} model")
        historical_data = await fetch_historical_data(request.stock_code, request.days)

        if historical_data is None or historical_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for stock symbol {request.stock_code}")

        if request.model.lower() == 'lstm':
            backtesting_results = backtest_lstm_model(request.stock_code, historical_data)
        elif request.model.lower() == 'arima':
            backtesting_results = backtest_arima_model(request.stock_code, historical_data)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid model specified: {request.model}")

        logger.info(f"Successfully completed {request.model.upper()} backtesting for {request.stock_code}")

        return backtesting_results
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error during {request.model.upper()} backtesting for {request.stock_code}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during backtesting: {str(e)}")

@app.get("/api/backtest/{stock_code}")
async def backtest_stock_get(stock_code: str, days: str = '1year', model: str = 'lstm'):
    """
    GET endpoint for backtesting. This is a wrapper around the POST endpoint for easier testing.
    """
    request = BacktestRequest(stock_code=stock_code, days=days, model=model)
    return await backtest_stock(request)


@app.get("/api/test/news/{symbol}")
async def test_fetch_news(symbol: str):
    """
    Fetch news articles related to a specific stock symbol.

    This endpoint retrieves the latest news articles for the given stock symbol, providing
    relevant information that may impact stock performance and investor sentiment.

    Parameters:
    - symbol (str): The stock symbol for which to fetch news articles (e.g., "RELIANCE").

    Returns:
    - dict: A dictionary containing the news articles for the specified stock symbol.

    Raises:
    - HTTPException: If an error occurs during the fetching of news articles (500 Internal Server Error).
    """
    try:
        news_data = await fetch_news_data(symbol)
        return {"news": news_data}
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


async def get_stock_analysis(stock_code: str):
    try:
        # Fetch necessary data
        news_data = await fetch_news_data(stock_code)
        historical_data = await fetch_historical_data(stock_code)

        if historical_data is None:
            raise HTTPException(status_code=404, detail=f"No historical data found for stock code {stock_code}")

        technical_indicators = {
            "sma_20": calculate_sma(historical_data['close'], 20),
            "ema_50": calculate_ema(historical_data['close'], 50),
            "rsi_14": calculate_rsi(historical_data['close'], 14),
            "macd": calculate_macd(historical_data['close']),
            "bollinger_bands": calculate_bollinger_bands(historical_data['close']),
            "atr": calculate_atr(historical_data['high'], historical_data['low'], historical_data['close'], 14)
        }

        lstm_prediction = await get_lstm_prediction(stock_code)

        gpt4_processor = GPT4Processor()
        news_sentiment, sentiment_explanation = gpt4_processor.analyze_news_sentiment(news_data)

        final_analysis = await gpt4_processor.final_analysis(
            stock_code,
            news_sentiment,
            sentiment_explanation,
            lstm_prediction,
            technical_indicators,
            historical_data
        )

        return {
            "stock_code": stock_code,
            "news_sentiment": news_sentiment,
            "sentiment_explanation": sentiment_explanation,
            "lstm_prediction": lstm_prediction,
            "technical_indicators": technical_indicators,
            "final_analysis": final_analysis
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error generating analysis for {stock_code}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating analysis for {stock_code}: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
