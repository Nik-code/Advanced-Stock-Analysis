import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import { getHistoricalData, getTechnicalIndicators, getQuote } from '../../services/api';
import './Dashboard.css';
import { ChartData, ChartOptions } from 'chart.js';

const Dashboard: React.FC = () => {
  const [stockCode, setStockCode] = useState<string>('');
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [technicalIndicators, setTechnicalIndicators] = useState<any>(null);
  const [quoteData, setQuoteData] = useState<any>(null);
  const [timeFrame, setTimeFrame] = useState<string>('1year');

  const handleSearch = async () => {
    if (stockCode) {
      try {
        const [historical, indicators, quote] = await Promise.all([
          getHistoricalData(stockCode, timeFrame),
          getTechnicalIndicators(stockCode),
          getQuote(`BSE:${stockCode}`)
        ]);
        setHistoricalData(historical);
        setTechnicalIndicators(indicators);
        setQuoteData(quote[`BSE:${stockCode}`]);
      } catch (error) {
        console.error('Error fetching stock data:', error);
      }
    }
  };

  const chartData: ChartData<'line'> = {
    labels: historicalData.map(data => data.date),
    datasets: [
      {
        type: 'line' as const,
        label: 'Close Price',
        data: historicalData.map(data => data.close),
        borderColor: 'rgb(75, 192, 192)',
        yAxisID: 'y',
      },
      {
        type: 'bar' as const,
        label: 'Volume',
        data: historicalData.map(data => data.volume),
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
        yAxisID: 'y1',
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    scales: {
      y: {
        type: 'linear' as const,
        display: true,
        position: 'left' as const,
      },
      y1: {
        type: 'linear' as const,
        display: true,
        position: 'right' as const,
        grid: {
          drawOnChartArea: false,
        },
      },
    },
  };

  return (
    <div className="dashboard">
      <h1>Stock Analysis Dashboard</h1>
      <div className="search-container">
        <input
          type="text"
          value={stockCode}
          onChange={(e) => setStockCode(e.target.value.toUpperCase())}
          placeholder="Enter stock code (e.g., TCS)"
          className="stock-search"
        />
        <button onClick={handleSearch} className="search-button">Search</button>
      </div>

      <div className="time-frame-selector">
        <button onClick={() => setTimeFrame('1month')}>1M</button>
        <button onClick={() => setTimeFrame('3months')}>3M</button>
        <button onClick={() => setTimeFrame('1year')}>1Y</button>
        <button onClick={() => setTimeFrame('5years')}>5Y</button>
      </div>

      {stockCode && quoteData && (
        <div className="stock-data">
          <h2>{stockCode} - {quoteData?.last_price}</h2>
          <p>Change: {quoteData?.change} ({quoteData?.change_percent}%)</p>

          <div className="chart-container">
            <h3>Historical Price Data</h3>
            <Line data={chartData} options={chartOptions} />
          </div>

          <div className="technical-indicators">
            <h3>Technical Indicators</h3>
            <ul>
              <li>SMA: {technicalIndicators?.indicators?.SMA_20[technicalIndicators.indicators.SMA_20.length - 1]?.value?.toFixed(2)}</li>
              <li>EMA: {technicalIndicators?.indicators?.EMA_20[technicalIndicators.indicators.EMA_20.length - 1]?.value?.toFixed(2)}</li>
              <li>RSI: {technicalIndicators?.indicators?.RSI[technicalIndicators.indicators.RSI.length - 1]?.value?.toFixed(2)}</li>
              <li>MACD: {technicalIndicators?.indicators?.MACD?.macd_line[technicalIndicators.indicators.MACD.macd_line.length - 1]?.value?.toFixed(2)}</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;