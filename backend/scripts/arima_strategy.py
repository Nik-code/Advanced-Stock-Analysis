import sys
import os
import asyncio
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import quantstats as qs
import logging
import matplotlib.pyplot as plt
from app.services.data_collection import fetch_historical_data
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ARIMAStrategy:
    def __init__(self, stock_code, initial_capital=100000, transaction_cost=0.001,
                 arima_order=(1, 1, 1), buy_threshold=0.01, sell_threshold=0.01):
        self.stock_code = stock_code
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.arima_order = arima_order
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.shares = 0
        self.trades = []
        self.portfolio_values = []

    async def run_strategy(self, historical_data):
        self.reset()
        if historical_data is None or len(historical_data) < 30:
            logger.warning(f"Insufficient data for {self.stock_code}")
            return None

        close_prices = historical_data['close'].values
        dates = pd.to_datetime(historical_data.index)  # Ensure dates are datetime

        try:
            model = ARIMA(close_prices, order=self.arima_order)
            model_fit = model.fit()
        except Exception as e:
            logger.error(f"Error fitting ARIMA model for {self.stock_code}: {str(e)}")
            return None

        for i in range(1, len(close_prices)):
            current_price = close_prices[i]
            prev_price = close_prices[i - 1]
            prediction = model_fit.forecast(steps=1)[0]

            if prediction > prev_price * (1 + self.buy_threshold) and self.cash > current_price:
                shares_to_buy = (self.cash * (1 - self.transaction_cost)) // current_price
                if shares_to_buy > 0:
                    self.shares += shares_to_buy
                    self.cash -= shares_to_buy * current_price * (1 + self.transaction_cost)
                    self.trades.append(('buy', shares_to_buy, current_price, dates[i]))
            elif prediction < prev_price * (1 - self.sell_threshold) and self.shares > 0:
                self.cash += self.shares * current_price * (1 - self.transaction_cost)
                self.trades.append(('sell', self.shares, current_price, dates[i]))
                self.shares = 0

            portfolio_value = self.cash + self.shares * current_price
            self.portfolio_values.append(portfolio_value)

        return pd.Series(self.portfolio_values, index=dates[1:])


def evaluate_strategy(strategy, returns, strategy_name):
    report = generate_detailed_report(strategy, returns, strategy_name)

    # Print the report to console
    print(report)

    # Save the report to a text file
    with open(f'{strategy_name.lower().replace(" ", "_")}_report.txt', 'w') as f:
        f.write(report)


def generate_detailed_report(strategy, returns, strategy_name):
    cumulative_returns = (1 + returns).cumprod() - 1
    total_return = cumulative_returns.iloc[-1]

    # Check if the index is DatetimeIndex, if not, assume it's a range index
    if isinstance(returns.index, pd.DatetimeIndex):
        days = (returns.index[-1] - returns.index[0]).days
    else:
        days = len(returns)  # Assume each data point represents a day

    annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0
    risk_free_rate = 0.02
    sharpe_ratio = qs.stats.sharpe(returns, rf=risk_free_rate, periods=252)
    max_drawdown = qs.stats.max_drawdown(returns)
    win_rate = qs.stats.win_rate(returns)
    sortino_ratio = qs.stats.sortino(returns, rf=risk_free_rate, periods=252)

    report = f"""
{strategy_name} Detailed Report
==============================

Performance Metrics:
-------------------
Total Return: {total_return:.2%}
Annualized Return: {annualized_return:.2%}
Sharpe Ratio: {sharpe_ratio:.2f}
Sortino Ratio: {sortino_ratio:.2f}
Max Drawdown: {max_drawdown:.2%}
Win Rate: {win_rate:.2%}

Trade Details:
--------------
"""

    total_profit = 0
    for trade in strategy.trades:
        action, amount, price, date = trade
        if action == 'buy':
            trade_value = -amount * price
        else:  # sell
            trade_value = amount * price
        total_profit += trade_value
        report += f"{date} - {action.upper()}: {amount} shares at ${price:.2f} (Trade Value: ${trade_value:.2f})\n"

    report += f"\nTotal Profit: ${total_profit:.2f}"
    report += f"\nFinal Portfolio Value: ${strategy.portfolio_values[-1]:.2f}"
    report += f"\nOverall Profit %: {(strategy.portfolio_values[-1] / strategy.initial_capital - 1) * 100:.2f}%"

    return report


async def run_strategy_test(stock_code, arima_order=(1, 1, 1), buy_threshold=0.01, sell_threshold=0.01, time_frame='1year'):
    strategy = ARIMAStrategy(stock_code, arima_order=arima_order, buy_threshold=buy_threshold,
                             sell_threshold=sell_threshold)

    historical_data = await fetch_historical_data(stock_code, time_frame)

    if historical_data is not None and len(historical_data) >= 30:
        portfolio_values = await strategy.run_strategy(historical_data)

        if portfolio_values is not None and len(portfolio_values) > 1:
            returns = portfolio_values.pct_change().dropna()
            strategy_name = f"ARIMA Strategy ({stock_code}, order={arima_order}, buy={buy_threshold}, sell={sell_threshold}, time_frame={time_frame})"
            evaluate_strategy(strategy, returns, strategy_name)
        else:
            print(f"Insufficient trading data for {stock_code}")
    else:
        print(f"Insufficient historical data for {stock_code}")


async def main():
    stocks_to_test = ['GAEL', 'INFY', 'TCS']  # Add more stocks as needed
    arima_orders = [(1, 1, 1), (2, 1, 2), (1, 1, 2)]  # Different ARIMA orders to test
    thresholds = [(0.01, 0.01), (0.02, 0.02), (0.015, 0.01)]  # Different buy/sell thresholds
    time_frames = ['1month', '3months', '1year']  # Available time frames

    all_reports = ""

    for stock in stocks_to_test:
        for order in arima_orders:
            for buy_thresh, sell_thresh in thresholds:
                for time_frame in time_frames:
                    strategy = ARIMAStrategy(stock, arima_order=order, buy_threshold=buy_thresh,
                                             sell_threshold=sell_thresh)
                    historical_data = await fetch_historical_data(stock, time_frame)

                    if historical_data is not None and len(historical_data) >= 30:
                        portfolio_values = await strategy.run_strategy(historical_data)

                        if portfolio_values is not None and len(portfolio_values) > 1:
                            returns = portfolio_values.pct_change().dropna()
                            strategy_name = f"ARIMA Strategy ({stock}, order={order}, buy={buy_thresh}, sell={sell_thresh}, time_frame={time_frame})"
                            report = generate_detailed_report(strategy, returns, strategy_name)
                            all_reports += report + "\n\n" + "=" * 50 + "\n\n"
                        else:
                            all_reports += f"Insufficient trading data for {stock} with time frame {time_frame}\n\n"
                    else:
                        all_reports += f"Insufficient historical data for {stock} with time frame {time_frame}\n\n"

    # Save all reports to a single text file
    with open('all_arima_strategy_reports.txt', 'w') as f:
        f.write(all_reports)

    print("All reports have been saved to 'all_arima_strategy_reports.txt'")


if __name__ == "__main__":
    asyncio.run(main())