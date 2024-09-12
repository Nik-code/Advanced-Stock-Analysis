from fastapi import APIRouter, HTTPException
from app.services.data_collection import fetch_historical_data, fetch_news_data
from app.services.zerodha_service import ZerodhaService
from app.services.technical_indicators import calculate_sma, calculate_ema, calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_atr
from app.models.backtesting import backtest_lstm_model, backtest_arima_model
import logging

router = APIRouter()
zerodha_service = ZerodhaService()
logger = logging.getLogger(__name__)


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


@router.get("/{symbol}/indicators")
async def get_technical_indicators(symbol: str, timeFrame: str = '1year'):
    try:
        historical_data = await fetch_historical_data(symbol, timeFrame)
        if historical_data is None:
            raise HTTPException(status_code=404, detail=f"No data found for stock symbol {symbol}")

        sma_20 = calculate_sma(historical_data['close'], 20)
        ema_50 = calculate_ema(historical_data['close'], 50)
        rsi_14 = calculate_rsi(historical_data['close'], 14)
        macd, signal, _ = calculate_macd(historical_data['close'])
        upper, middle, lower = calculate_bollinger_bands(historical_data['close'])
        atr = calculate_atr(historical_data['high'], historical_data['low'], historical_data['close'], 14)

        return {
            "sma_20": sma_20.tolist(),
            "ema_50": ema_50.tolist(),
            "rsi_14": rsi_14.tolist(),
            "macd": macd.tolist(),
            "macd_signal": signal.tolist(),
            "bollinger_upper": upper.tolist(),
            "bollinger_middle": middle.tolist(),
            "bollinger_lower": lower.tolist(),
            "atr": atr.tolist(),
            "dates": historical_data['date'].tolist(),
            "close_prices": historical_data['close'].tolist()
        }
    except Exception as e:
        logger.error(f"Error calculating technical indicators for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/realtime")
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
