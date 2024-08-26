import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    return data.rolling(window=window).mean()

def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    return data.ewm(span=window, adjust=False).mean()

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    delta = data.diff()
    gain, loss = delta.copy(), delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = -loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    try:
        ema_fast = calculate_ema(data, fast_period)
        ema_slow = calculate_ema(data, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = calculate_ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    except Exception as e:
        logger.error(f"Error in calculate_macd: {str(e)}")
        raise


def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()