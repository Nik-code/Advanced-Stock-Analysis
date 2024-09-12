from fastapi import APIRouter, HTTPException
from app.services.data_collection import fetch_historical_data, fetch_news_data
from app.services.zerodha_service import ZerodhaService
from app.services.technical_indicators import calculate_sma, calculate_ema, calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_atr
from app.models.backtesting import backtest_lstm_model, backtest_arima_model
from app.services.llm_integration import GPT4Processor
import logging
import os
import joblib
import numpy as np
from app.models.lstm_model import LSTMStockPredictor

router = APIRouter()
zerodha_service = ZerodhaService()
logger = logging.getLogger(__name__)
llm_processor = GPT4Processor()

@router.get("/quote")
async def get_quote(instruments: str):
    try:
        quote_data = zerodha_service.get_quote(instruments)
        if quote_data is None:
            raise HTTPException(status_code=404, detail="Quote not found")
        return quote_data
    except Exception as e:
        logger.error(f"Error fetching quote for {instruments}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching quote")

@router.get("/live/{stock_code}")
async def get_live_stock_data(stock_code: str):
    try:
        instrument_token = zerodha_service.get_instrument_token("BSE", stock_code)
        if not instrument_token:
            raise HTTPException(status_code=404, detail=f"No instrument token found for stock code {stock_code}")
        live_data = zerodha_service.get_quote([instrument_token])
        if live_data is None or str(instrument_token) not in live_data:
            raise HTTPException(status_code=404, detail=f"No live data found for stock code {stock_code}")
        return live_data[str(instrument_token)]
    except Exception as e:
        logger.error(f"Error fetching live data for {stock_code}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical/{stock_code}")
async def get_historical_stock_data(stock_code: str, timeframe: str = '1year'):
    try:
        historical_data = await fetch_historical_data(stock_code, timeframe)
        if historical_data is None:
            raise HTTPException(status_code=404, detail=f"No historical data found for stock code {stock_code}")
        return historical_data.to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error fetching historical data for {stock_code}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news/{stock_code}")
async def get_stock_news(stock_code: str):
    try:
        news_data = await fetch_news_data(stock_code)
        return {"news": news_data}
    except Exception as e:
        logger.error(f"Error fetching news for {stock_code}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/{stock_code}")
async def predict_stock(stock_code: str):
    try:
        logger.info(f"Received prediction request for {stock_code}")
        model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        lstm_model_path = os.path.join(model_dir, f'{stock_code}_lstm_model.h5')
        scaler_path = os.path.join(model_dir, f'{stock_code}_scaler.pkl')
        arima_model_path = os.path.join(model_dir, f'{stock_code}_arima_model.pkl')

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

@router.get("/analysis/{stock_code}")
async def get_stock_analysis(stock_code: str):
    try:
        historical_data = await fetch_historical_data(stock_code, '1year')
        news_data = await fetch_news_data(stock_code)
        lstm_prediction = await predict_stock(stock_code)
        arima_prediction = lstm_prediction  # We're using the same prediction endpoint for both LSTM and ARIMA

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

@router.get("/top_stocks")
async def get_top_stocks():
    try:
        # Implement logic to analyze all stocks and return top picks
        top_stocks = await analyze_all_stocks()
        return {"top_stocks": top_stocks}
    except Exception as e:
        logger.error(f"Error getting top stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def analyze_all_stocks():
    # Implement the logic to analyze all stocks and return top picks
    # This is a placeholder implementation
    return ["RELIANCE", "TCS", "HDFCBANK"]  # Replace with actual analysis logic
