import React, { useState, useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartData,
  ChartOptions,
} from 'chart.js';
import { Chart as ReactChart } from 'react-chartjs-2';
import { getHistoricalData, getTechnicalIndicators, getQuote } from '../../services/api';
import './Dashboard.css';
import StockOverview from '../../components/StockOverview/StockOverview';
import TechnicalAnalysis from '../../components/TechnicalAnalysis/TechnicalAnalysis';
import VolumeAnalysis from '../../components/VolumeAnalysis/VolumeAnalysis';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const Dashboard: React.FC = () => {
  const [stockCode, setStockCode] = useState<string>('');
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [technicalIndicators, setTechnicalIndicators] = useState<any>(null);
  const [quoteData, setQuoteData] = useState<any>(null);
  const [timeFrame, setTimeFrame] = useState<string>('1year');

  const chartRef = useRef<ChartJS<'line' | 'bar'> | null>(null);

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

  useEffect(() => {
    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
      }
    };
  }, []);

  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.update();
    }
  }, [historicalData]);

  const chartData: ChartData<'line' | 'bar', number[], string> = {
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

  const chartOptions: ChartOptions<'line' | 'bar'> = {
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
          <StockOverview stockCode={stockCode} quoteData={quoteData} />

          <div className="chart-container">
            <h3>Historical Price and Volume Data</h3>
            <ReactChart
              ref={chartRef}
              type='line'
              data={chartData}
              options={chartOptions}
            />
          </div>

          <TechnicalAnalysis technicalIndicators={technicalIndicators} />
          <VolumeAnalysis historicalData={historicalData} />
        </div>
      )}
    </div>
  );
};

export default Dashboard;
