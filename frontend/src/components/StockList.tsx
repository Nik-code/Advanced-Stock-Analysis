import React, { useEffect, useState } from 'react';
import { getMarketOverview, getQuote } from '../services/api';
import './StockList.css';

interface Stock {
  symbol: string;
  last_price: number;
  change: number;
  change_percent: number;
}

const StockList: React.FC = () => {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStocks = async () => {
      try {
        const overviewData = await getMarketOverview(20); // Fetch top 20 stocks
        const stockSymbols = overviewData.map((stock: Stock) => `BSE:${stock.symbol}`).join(',');
        const quoteData = await getQuote(stockSymbols);

        const combinedData = overviewData.map((stock: Stock) => ({
          ...stock,
          ...quoteData[`BSE:${stock.symbol}`]
        }));

        setStocks(combinedData);
        setLoading(false);
      } catch (err) {
        setError('Failed to fetch stocks. Please try again later.');
        setLoading(false);
      }
    };
    fetchStocks();
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div className="stockList">
      <h2>Stock List</h2>
      <ul>
        {stocks.map((stock) => (
          <li key={stock.symbol}>
            {stock.symbol}: ₹{stock.last_price.toFixed(2)} (
            <span style={{ color: stock.change >= 0 ? 'green' : 'red' }}>
              {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)} ({stock.change_percent.toFixed(2)}%)
            </span>
            )
          </li>
        ))}
      </ul>
    </div>
  );
};

export default StockList;