// src/components/StockDetail.tsx
import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getHistoricalData, getTechnicalIndicators } from '../services/api';
import './StockDetail.css';

interface HistoricalData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface TechnicalIndicators {
  indicators: {
    sma_20: { date: string; value: number }[];
    ema_50: { date: string; value: number }[];
    rsi_14: { date: string; value: number }[];
    macd: { date: string; value: number }[];
    atr: { date: string; value: number }[];
  };
}

const StockDetail: React.FC = () => {
  const { symbol } = useParams<{ symbol: string }>();
  const [historicalData, setHistoricalData] = useState<HistoricalData[]>([]);
  const [technicalIndicators, setTechnicalIndicators] = useState<TechnicalIndicators | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

useEffect(() => {
  const fetchData = async () => {
    try {
      const [historicalResponse, indicatorsResponse] = await Promise.all([
        getHistoricalData(symbol as string),
        getTechnicalIndicators(symbol as string, '1year')
      ]);

      setHistoricalData(historicalResponse);
      setTechnicalIndicators(indicatorsResponse); // This should now work correctly
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch stock data. Please try again later.');
      setLoading(false);
    }
  };

  if (symbol) {
    fetchData();
  }
}, [symbol]);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>{error}</div>;

  return (
      <div className="stockDetail">
        <h2>{symbol} Stock Details</h2>
        <div className="technicalIndicators">
          <h3>Technical Indicators</h3>
          {technicalIndicators && technicalIndicators.indicators && (
            <ul>
              <li>SMA 20: {technicalIndicators.indicators.sma_20[technicalIndicators.indicators.sma_20.length - 1]?.value.toFixed(2) || 'N/A'}</li>
              <li>EMA 50: {technicalIndicators.indicators.ema_50[technicalIndicators.indicators.ema_50.length - 1]?.value.toFixed(2) || 'N/A'}</li>
              <li>RSI: {technicalIndicators.indicators.rsi_14[technicalIndicators.indicators.rsi_14.length - 1]?.value.toFixed(2) || 'N/A'}</li>
              <li>MACD: {technicalIndicators.indicators.macd[technicalIndicators.indicators.macd.length - 1]?.value.toFixed(2) || 'N/A'}</li>
              <li>ATR: {technicalIndicators.indicators.atr[technicalIndicators.indicators.atr.length - 1]?.value.toFixed(2) || 'N/A'}</li>
            </ul>
          )}
        </div>
        <div className="historicalData">
          <h3>Historical Data</h3>
          <table>
            <thead>
            <tr>
              <th>Date</th>
              <th>Open</th>
              <th>High</th>
              <th>Low</th>
              <th>Close</th>
              <th>Volume</th>
            </tr>
            </thead>
            <tbody>
            {historicalData.slice(0, 10).map((data) => (
                <tr key={data.date}>
                  <td>{data.date}</td>
                  <td>{data.open.toFixed(2)}</td>
                  <td>{data.high.toFixed(2)}</td>
                  <td>{data.low.toFixed(2)}</td>
                  <td>{data.close.toFixed(2)}</td>
                  <td>{data.volume}</td>
                </tr>
            ))}
            </tbody>
          </table>
        </div>
      </div>
  );
};

export default StockDetail;