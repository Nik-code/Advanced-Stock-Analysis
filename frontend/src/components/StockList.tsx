import React, { useEffect, useState } from 'react';
import { getStocks } from '../services/mockDataService';
import './StockList.css';

interface Stock {
  symbol: string;
  name: string;
  price: number;
  change: number;
}

const StockList: React.FC = () => {
  const [stocks, setStocks] = useState<Stock[]>([]);

  useEffect(() => {
    const fetchStocks = async () => {
      const stockData = await getStocks();
      setStocks(stockData);
    };
    fetchStocks();
  }, []);

  return (
    <div className="stockList">
      <h2>Stock List</h2>
      <ul>
        {stocks.map((stock) => (
          <li key={stock.symbol}>
            {stock.name} ({stock.symbol}): ₹{stock.price.toFixed(2)} ({stock.change > 0 ? '+' : ''}{stock.change.toFixed(2)})
          </li>
        ))}
      </ul>
    </div>
  );
};

export default StockList;