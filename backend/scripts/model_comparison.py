import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models.lstm_model import LSTMStockPredictor
from app.models.arima_model import ARIMAStockPredictor
from app.services.data_collection import fetch_historical_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import asyncio
import joblib
import logging
from app.models.backtesting import backtest_model as backtesting_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INVESTMENT_AMOUNT = 100000  # Initial investment amount
TRANSACTION_COST = 0.001  # 0.1% transaction cost
THRESHOLD = 0.01  # 1% threshold for buy/sell decisions
MAX_STOCKS = 10
BATCH_SIZE = 3
RATE_LIMIT_DELAY = 1  # 1 second delay between batches


async def backtest_model(model_type, stock_code, historical_data):
    if len(historical_data) < 61:
        logger.warning(f"Insufficient data for {stock_code} to perform backtesting. Skipping.")
        return None

    try:
        result = backtesting_model(stock_code, historical_data, model_type, 
                                   investment_amount=INVESTMENT_AMOUNT, 
                                   threshold=THRESHOLD, 
                                   transaction_cost=TRANSACTION_COST)
        return result
    except Exception as e:
        logger.error(f"Error in {model_type} backtesting for {stock_code}: {str(e)}")
        return None


async def process_batch(batch, time_frame):
    results = []
    insufficient_data = []
    for stock_code in batch:
        logger.info(f"Backtesting {stock_code}")
        historical_data = await fetch_historical_data(stock_code, time_frame)
        if historical_data is None or len(historical_data) < 61:
            logger.warning(f"Insufficient data available for {stock_code}. Got {len(historical_data) if historical_data is not None else 0} data points, need at least 61.")
            insufficient_data.append(stock_code)
            continue

        lstm_result = await backtest_model('LSTM', stock_code, historical_data)
        arima_result = await backtest_model('ARIMA', stock_code, historical_data)

        if lstm_result and arima_result:
            results.append({
                'stock_code': stock_code,
                'lstm_result': lstm_result,
                'arima_result': arima_result
            })
        else:
            logger.warning(f"Skipping {stock_code} due to backtesting failure")

    return results, insufficient_data


async def compare_models(stock_codes, time_frame='3months'):
    all_results = []
    all_insufficient_data = []

    for i in range(0, len(stock_codes), BATCH_SIZE):
        batch = stock_codes[i:i+BATCH_SIZE]
        results, insufficient_data = await process_batch(batch, time_frame)
        all_results.extend(results)
        all_insufficient_data.extend(insufficient_data)

        if i + BATCH_SIZE < len(stock_codes):
            logger.info(f"Waiting {RATE_LIMIT_DELAY} second(s) before processing next batch...")
            await asyncio.sleep(RATE_LIMIT_DELAY)

    return all_results, all_insufficient_data


def print_summary(results, insufficient_data):
    print("\nModel Comparison Summary:")
    print("=" * 50)

    if not results:
        print("No stocks had sufficient data for backtesting.")
        return

    lstm_results = []
    arima_results = []
    lstm_profitable = 0
    arima_profitable = 0

    for result in results:
        print(f"\nStock: {result['stock_code']}")
        print("-" * 30)
        for model in ['lstm_result', 'arima_result']:
            model_result = result[model]
            print(f"{model_result['model_type']} Model:")
            print(f"  Initial Investment: ${model_result['initial_investment']:.2f}")
            print(f"  Final Portfolio Value: ${model_result['final_portfolio_value']:.2f}")
            print(f"  Total Return: {model_result['total_return_percentage']:.2f}%")
            print(f"  Number of Trades: {model_result['number_of_trades']}")
            
            if model == 'lstm_result':
                lstm_results.append((result['stock_code'], model_result['total_return_percentage']))
                if model_result['total_return_percentage'] > 0:
                    lstm_profitable += 1
            else:
                arima_results.append((result['stock_code'], model_result['total_return_percentage']))
                if model_result['total_return_percentage'] > 0:
                    arima_profitable += 1
        print()

    # Calculate overall performance
    lstm_total_return = sum(r[1] for r in lstm_results)
    arima_total_return = sum(r[1] for r in arima_results)
    lstm_avg_return = lstm_total_return / len(results)
    arima_avg_return = arima_total_return / len(results)

    print("Overall Performance:")
    print(f"LSTM Average Return: {lstm_avg_return:.2f}%")
    print(f"ARIMA Average Return: {arima_avg_return:.2f}%")

    # Best and worst performing stocks
    lstm_results.sort(key=lambda x: x[1], reverse=True)
    arima_results.sort(key=lambda x: x[1], reverse=True)

    print("\nBest Performing Stocks:")
    print(f"LSTM: {lstm_results[0][0]} ({lstm_results[0][1]:.2f}%)")
    print(f"ARIMA: {arima_results[0][0]} ({arima_results[0][1]:.2f}%)")

    print("\nWorst Performing Stocks:")
    print(f"LSTM: {lstm_results[-1][0]} ({lstm_results[-1][1]:.2f}%)")
    print(f"ARIMA: {arima_results[-1][0]} ({arima_results[-1][1]:.2f}%)")

    print("\nProfitability:")
    print(f"LSTM: Profitable on {lstm_profitable} out of {len(results)} stocks ({lstm_profitable/len(results)*100:.2f}%)")
    print(f"ARIMA: Profitable on {arima_profitable} out of {len(results)} stocks ({arima_profitable/len(results)*100:.2f}%)")

    print("\nTop 5 Performing Stocks for Each Model:")
    print("LSTM:")
    for stock, return_percentage in lstm_results[:5]:
        print(f"  {stock}: {return_percentage:.2f}%")
    print("ARIMA:")
    for stock, return_percentage in arima_results[:5]:
        print(f"  {stock}: {return_percentage:.2f}%")

    print("\nBottom 5 Performing Stocks for Each Model:")
    print("LSTM:")
    for stock, return_percentage in reversed(lstm_results[-5:]):
        print(f"  {stock}: {return_percentage:.2f}%")
    print("ARIMA:")
    for stock, return_percentage in reversed(arima_results[-5:]):
        print(f"  {stock}: {return_percentage:.2f}%")

    print("\nStocks with insufficient data:")
    for stock in insufficient_data:
        print(f"- {stock}")

    print(f"\nTotal stocks processed: {len(results)}")
    print(f"Stocks with insufficient data: {len(insufficient_data)}")


def get_trained_stock_codes():
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    stock_codes = set()
    for filename in os.listdir(model_dir):
        if filename.endswith('_lstm_model.h5'):
            stock_code = filename.split('_lstm_model.h5')[0]
            stock_codes.add(stock_code)
    return list(stock_codes)[:MAX_STOCKS]  # Limit to MAX_STOCKS


if __name__ == "__main__":
    stock_codes = get_trained_stock_codes()
    logger.info(f"Found {len(stock_codes)} trained models (limited to {MAX_STOCKS}): {', '.join(stock_codes)}")

    results, insufficient_data = asyncio.run(compare_models(stock_codes))
    print_summary(results, insufficient_data)
