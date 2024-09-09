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
from app.models.backtesting import backtest_lstm_model, backtest_arima_model
from app.models.lstm_model import LSTMStockPredictor
from app.services.data_collection import fetch_historical_data, fetch_process_store_data, update_influxdb_with_latest_data, fetch_news_data, process_news_data
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

app.include_router(stocks.router, prefix="/api")


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


@app.get("/api/historical/{code}")
async def get_historical_data(code: str, timeFrame: str = '1year'):
    """
    Fetch historical data for a given stock code.

    Parameters:
    - code (str): The stock code for which historical data is requested.
    - timeFrame (str): The time frame for the historical data. 
                       It can be one of the following values:
                       - '1month'
                       - '3months'
                       - '1year'
                       - '5years'

    Returns:
    - List[dict]: A list of historical data records for the specified stock code. Each record contains:
        - date (str): The date of the record in ISO 8601 format.
        - open (float): The opening price of the stock on that date.
        - high (float): The highest price of the stock on that date.
        - low (float): The lowest price of the stock on that date.
        - close (float): The closing price of the stock on that date.
        - volume (int): The trading volume of the stock on that date.

    Example of a successful response:
    [
        {
            "date": "2024-08-09T00:00:00+05:30",
            "open": 4227.95,
            "high": 4251.5,
            "low": 4215.0,
            "close": 4230.05,
            "volume": 169602
        },
        ...
    ]

    Raises:
    - HTTPException: If no data is found for the stock code or if an internal error occurs.
    """
    try:
        data = await fetch_historical_data(code, timeFrame)
        if data is None:
            raise HTTPException(status_code=404, detail=f"No data found for stock code {code}")
        return data.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error fetching historical data for {code}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again later.")


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


@app.get("/api/stocks/{symbol}/indicators")
async def get_technical_indicators(symbol: str, timeFrame: str = '1year'):
    """
    Retrieve technical indicators for a specified stock symbol over a given time frame.

    This endpoint calculates various technical indicators based on historical stock data, including:
    - Simple Moving Average (SMA)
    - Exponential Moving Average (EMA)
    - Relative Strength Index (RSI)
    - Moving Average Convergence Divergence (MACD)
    - Bollinger Bands
    - Average True Range (ATR)

    Parameters:
    - symbol (str): The stock symbol for which to retrieve technical indicators (e.g., "RELIANCE").
    - timeFrame (str): The time frame for historical data (default is '1year').

    Returns:
    - dict: A dictionary containing the calculated technical indicators and historical data, structured as follows:
        {
            "sma_20": List[float],  # 20-period SMA values
            "ema_50": List[float],  # 50-period EMA values
            "rsi_14": List[float],   # 14-period RSI values
            "macd": List[float],     # MACD values
            "macd_signal": List[float],  # MACD signal line values
            "bollinger_upper": List[float],  # Upper Bollinger Band values
            "bollinger_middle": List[float],  # Middle Bollinger Band values
            "bollinger_lower": List[float],  # Lower Bollinger Band values
            "atr": List[float],      # Average True Range values
            "dates": List[str],      # Dates corresponding to the historical data
            "close_prices": List[float]  # Closing prices for the historical data
        }

    Raises:
    - HTTPException: If no historical data is found for the specified stock symbol (404 Not Found).
    - HTTPException: If an error occurs during the calculation of technical indicators (500 Internal Server Error).
    """
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
    """
    Fetch real-time market data for a specified stock symbol.

    This endpoint retrieves the latest trading information for the given stock symbol, including
    instrument token, last price, volume, and order book depth.

    Parameters:
    - symbol (str): The stock symbol for which to fetch real-time data (e.g., "RELIANCE").

    Returns:
    - dict: A dictionary containing real-time data for the specified stock symbol, structured as follows:
    {
        "instrument_token": int,  # Unique identifier for the stock instrument
        "timestamp": str,         # Timestamp of the last update
        "last_trade_time": str,   # Time of the last trade
        "last_price": float,      # Last traded price
        "last_quantity": int,     # Quantity of the last trade
        "buy_quantity": int,      # Total quantity available for buying
        "sell_quantity": int,     # Total quantity available for selling
        "volume": int,            # Total volume traded
        "average_price": float,   # Average price of trades
        "oi": int,                # Open interest
        "oi_day_high": float,     # Highest open interest for the day
        "oi_day_low": float,      # Lowest open interest for the day
        "net_change": float,      # Net change in price
        "lower_circuit_limit": float,  # Lower circuit limit price
        "upper_circuit_limit": float,  # Upper circuit limit price
        "ohlc": {                 # Open, High, Low, Close data
            "open": float,        # Opening price
            "high": float,        # Highest price
            "low": float,         # Lowest price
            "close": float        # Closing price
        },
        "depth": {                # Order book depth
            "buy": [              # List of buy orders
                {
                    "price": float,  # Price of the buy order
                    "quantity": int, # Quantity of the buy order
                    "orders": int    # Number of orders at this price
                },
                ...
            ],
            "sell": [             # List of sell orders
                {
                    "price": float,  # Price of the sell order
                    "quantity": int, # Quantity of the sell order
                    "orders": int    # Number of orders at this price
                },
                ...
            ]
        }
    }

    Raises:
    - HTTPException: If no instrument token is found for the stock symbol (404 Not Found).
    - HTTPException: If no real-time data is found for the stock symbol (404 Not Found).
    - HTTPException: If an error occurs during the fetching of real-time data (500 Internal Server Error).
    """
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


class PredictionRequest(BaseModel):
    data: List[float]


@app.post("/api/predict/{stock_code}") # TODO: Not functioning at all
async def predict_stock(stock_code: str, request: PredictionRequest):
    """
    Generate stock price predictions using LSTM and ARIMA models.

    This function takes a stock code and historical price data, then uses pre-trained
    LSTM and ARIMA models to generate future price predictions. It also calculates
    past predictions for comparison.

    Args:
        stock_code (str): The code or symbol of the stock to predict.
        request (PredictionRequest): A request object containing historical closing prices for the stock.

    Returns:
        dict: A dictionary containing the following keys:
            - lstm_predictions: Future predictions from the LSTM model.
            - arima_predictions: Future predictions from the ARIMA model.
            - ensemble_predictions: Combined predictions from both models.
            - confidence_intervals: Upper and lower bounds for predictions.
            - past_predictions_lstm: Past predictions from the LSTM model.
            - past_predictions_arima: Past predictions from the ARIMA model.

    Raises:
        HTTPException: If there's an error during the prediction process.
    """
    try:
        logger.info(f"Received prediction request for {stock_code}")
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        lstm_model_path = os.path.join(model_dir, f'{stock_code}_lstm_model.h5')
        arima_model_path = os.path.join(model_dir, f'{stock_code}_arima_model.pkl')
        scaler_path = os.path.join(model_dir, f'{stock_code}_scaler.pkl')

        if not os.path.exists(lstm_model_path) or not os.path.exists(arima_model_path) or not os.path.exists(scaler_path):
            logger.error(f"Models or scaler not found for {stock_code}")
            raise HTTPException(status_code=404, detail=f"Models not found for {stock_code}")

        lstm_predictor = LSTMStockPredictor(input_shape=(60, 1))
        lstm_predictor.load_model(lstm_model_path)
        arima_predictor = joblib.load(arima_model_path)
        scaler = joblib.load(scaler_path)

        data = np.array(request.data).reshape(-1, 1)
        scaled_data = scaler.transform(data)

        # Generate past predictions using a rolling window
        past_predictions_lstm = []
        past_predictions_arima = []
        for i in range(60, len(scaled_data)):
            X = scaled_data[i-60:i].reshape(1, 60, 1)
            lstm_prediction = lstm_predictor.predict(X)
            arima_prediction = arima_predictor.predict(1)
            past_predictions_lstm.append(lstm_prediction[0][0])
            past_predictions_arima.append(arima_prediction[0])
        
        past_predictions_lstm = scaler.inverse_transform(np.array(past_predictions_lstm).reshape(-1, 1))
        past_predictions_arima = scaler.inverse_transform(np.array(past_predictions_arima).reshape(-1, 1))
        
        # Generate future predictions with confidence intervals
        future_predictions_lstm = []
        future_predictions_arima = []
        confidence_intervals = []
        num_simulations = 100
        forecast_horizon = 7
        last_sequence = scaled_data[-60:].reshape(1, 60, 1)

        for _ in range(forecast_horizon):
            simulations_lstm = []
            for _ in range(num_simulations):
                lstm_prediction = lstm_predictor.predict(last_sequence)
                simulations_lstm.append(lstm_prediction[0][0])
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = lstm_prediction[0][0]

            arima_prediction = arima_predictor.predict(1)

            mean_prediction_lstm = np.mean(simulations_lstm)
            ci_lower = np.percentile(simulations_lstm, 5)
            ci_upper = np.percentile(simulations_lstm, 95)

            future_predictions_lstm.append(mean_prediction_lstm)
            future_predictions_arima.append(arima_prediction[0])
            confidence_intervals.append((ci_lower, ci_upper))

        future_predictions_lstm = scaler.inverse_transform(np.array(future_predictions_lstm).reshape(-1, 1))
        future_predictions_arima = scaler.inverse_transform(np.array(future_predictions_arima).reshape(-1, 1))
        confidence_intervals = scaler.inverse_transform(np.array(confidence_intervals))

        # Combine LSTM and ARIMA predictions (simple average)
        ensemble_predictions = (future_predictions_lstm + future_predictions_arima) / 2

        logger.info(f"Successfully generated predictions for {stock_code}")

        return {
            "lstm_predictions": future_predictions_lstm.flatten().tolist(),
            "arima_predictions": future_predictions_arima.flatten().tolist(),
            "ensemble_predictions": ensemble_predictions.flatten().tolist(),
            "confidence_intervals": confidence_intervals.tolist(),
            "past_predictions_lstm": past_predictions_lstm.flatten().tolist(),
            "past_predictions_arima": past_predictions_arima.flatten().tolist()
        }
    except Exception as e:
        logger.error(f"Error making prediction for {stock_code}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction for {stock_code}: {str(e)}")


@app.post("/api/backtest/{stock_code}") # TODO: Gives bad predictions
async def backtest_stock(stock_code: str, days: str = '1year'):
    """
    Backtest the LSTM model for a given stock over a specified time period.

    This endpoint allows users to perform backtesting on the LSTM model using historical data for a specific stock. 
    The backtesting results will provide insights into the model's performance based on past data.

    Parameters:
    - stock_code (str): The stock symbol for which backtesting is requested (e.g., "RELIANCE").
    - days (str): The time frame for historical data. Default is '1year'. 
                  It can be one of the following values:
                  - '1month'
                  - '3months'
                  - '1year'
                  - '5years'

    Returns:
    - dict: A dictionary containing the backtesting results, which may include metrics such as:
        - total_return: The total return from the backtesting period.
        - number_of_trades: The total number of trades executed during the backtesting.
        - trades: A list of trades made during the backtesting period, including buy/sell actions and prices.

    Raises:
    - HTTPException: If no historical data is found for the specified stock symbol (404 Not Found).
    - HTTPException: If an internal error occurs during the backtesting process (500 Internal Server Error).
    """
    try:
        logger.info(f"Received backtesting request for {stock_code}")
        historical_data = await fetch_historical_data(stock_code, days)

        if historical_data is None:
            raise HTTPException(status_code=404, detail=f"No data found for stock symbol {stock_code}")

        backtesting_results = backtest_lstm_model(stock_code, historical_data)

        logger.info(f"Successfully completed backtesting for {stock_code}")

        return backtesting_results
    except Exception as e:
        logger.error(f"Error during backtesting for {stock_code}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during backtesting for {stock_code}: {str(e)}")


@app.post("/api/backtest_arima/{stock_code}")
async def backtest_stock_arima(stock_code: str, days: str = '1year'):
    """
    Backtest the ARIMA model for a given stock over a specified time period.

    This endpoint allows users to perform backtesting on the ARIMA model using historical data for a specific stock. 
    The backtesting results will provide insights into the model's performance based on past data.

    Parameters:
    - stock_code (str): The stock symbol for which backtesting is requested (e.g., "RELIANCE").
    - days (str): The time frame for historical data. Default is '1year'.
                  It can be one of the following values:
                  - '1month'
                  - '3months'
                  - '1year'
                  - '5years'

    Returns:
    - dict: A dictionary containing the backtesting results, which may include metrics such as:
        - total_return: The total return from the backtesting period.
        - number_of_trades: The total number of trades executed during the backtesting.
        - trades: A list of trades made during the backtesting period, including buy/sell actions and prices.

    Raises:
    - HTTPException: If no historical data is found for the specified stock symbol (404 Not Found).
    - HTTPException: If an internal error occurs during the backtesting process (500 Internal Server Error).
    """
    try:
        logger.info(f"Received ARIMA backtesting request for {stock_code}")
        historical_data = await fetch_historical_data(stock_code, days)

        if historical_data is None:
            raise HTTPException(status_code=404, detail=f"No data found for stock symbol {stock_code}")

        backtesting_results = backtest_arima_model(stock_code, historical_data)

        logger.info(f"Successfully completed ARIMA backtesting for {stock_code}")

        return backtesting_results
    except Exception as e:
        logger.error(f"Error during ARIMA backtesting for {stock_code}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during ARIMA backtesting for {stock_code}: {str(e)}")


@app.get("/api/sentiment/{symbol}")  # TODO: Doesn't work
async def get_sentiment(symbol: str):
    try:
        logger.info(f"Fetching news data for {symbol}")
        news_data = await fetch_news_data(symbol)
        logger.info(f"Fetched {len(news_data)} news articles for {symbol}")

        # Get LSTM prediction
        historical_data = await fetch_historical_data(symbol, '1year')
        if historical_data is None:
            raise HTTPException(status_code=404, detail=f"No data found for stock symbol {symbol}")
        close_prices = historical_data['close'].tolist()
        lstm_prediction = await predict_stock(symbol, close_prices)

        logger.info(f"Processing news data for {symbol}")
        news_analysis = await process_news_data(news_data, lstm_prediction['predictions'][0], symbol, historical_data)
        logger.info(f"Sentiment analysis result for {symbol}: {news_analysis}")
        return {
            "sentiment": news_analysis['sentiment'],
            "explanation": news_analysis['explanation'],
            "analysis": news_analysis['analysis'],
            "lstm_prediction": lstm_prediction['predictions'][0]
        }
    except Exception as e:
        logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test/news/{symbol}")
async def test_fetch_news(symbol: str):
    """
    Fetch news articles related to a specific stock symbol.

    This endpoint retrieves the latest news articles for the given stock symbol, providing
    relevant information that may impact stock performance and investor sentiment.

    Parameters:
    - symbol (str): The stock symbol for which to fetch news articles (e.g., "RELIANCE").

    Returns:
    - dict: A dictionary containing a list of news articles structured as follows:
    {
        "news": [
            {
                "title": str,            # Title of the news article
                "link": str,             # URL link to the full article
                "published_time": str,   # Publication time of the article in RFC 2822 format
                "summary": str           # Summary or excerpt of the article
            },
            ...
        ]
    }

    Raises:
    - HTTPException: If an error occurs during the fetching of news articles (500 Internal Server Error).
    """
    try:
        news_data = await fetch_news_data(symbol)
        return {"news": news_data}
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
