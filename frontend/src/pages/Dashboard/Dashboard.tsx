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
  TimeScale,
  ChartData,
  ChartOptions
} from 'chart.js';
import { Chart as ReactChart } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
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
  Legend,
  TimeScale
);

const Dashboard: React.FC = () => {
  const [darkMode, setDarkMode] = useState(false);
  const [stockCode, setStockCode] = useState<string>('');
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [technicalIndicators, setTechnicalIndicators] = useState<any>(null);
  const [quoteData, setQuoteData] = useState<any>(null);
  const [timeFrame, setTimeFrame] = useState<string>('1year');

  const chartRef = useRef<ChartJS<'line' | 'bar'> | null>(null);

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

  const handleSearch = async (newTimeFrame?: string) => {
    console.log('Searching for stock:', stockCode);
    if (stockCode) {
      try {
        const timeFrameToUse = newTimeFrame || timeFrame;
        const [historical, indicators, quote] = await Promise.all([
          getHistoricalData(stockCode, timeFrameToUse),
          getTechnicalIndicators(stockCode),
          getQuote(`BSE:${stockCode}`)
        ]);
        console.log('Historical data:', historical);
        console.log('Technical indicators:', indicators);
        console.log('Quote data:', quote);
        setHistoricalData(historical);
        setTechnicalIndicators(indicators);
        setQuoteData(quote[`BSE:${stockCode}`]);
        if (newTimeFrame) {
          setTimeFrame(newTimeFrame);
        }
      } catch (error) {
        console.error('Error fetching stock data:', error);
      }
    }
  };

  const handleSearchClick = () => {
    handleSearch();
  };

  useEffect(() => {
    document.body.classList.toggle('dark-mode', darkMode);
  }, [darkMode]);

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

  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.data = chartData;
      chartRef.current.update();
    }
  }, [historicalData, chartData]);

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
    <div className={`dashboard ${darkMode ? 'dark-mode' : ''}`}>
      <header>
        <h1>Advanced Stock Analysis</h1>
        <button onClick={() => setDarkMode(!darkMode)}>
          {darkMode ? 'Light Mode' : 'Dark Mode'}
        </button>
      </header>
      <div className="dashboard-content">
        <aside className="sidebar">
          <div className="search-container">
            <input
              type="text"
              value={stockCode}
              onChange={(e) => setStockCode(e.target.value.toUpperCase())}
              placeholder="Enter stock code (e.g., TCS)"
              className="stock-search"
            />
            <button onClick={handleSearchClick} className="search-button">Search</button>
          </div>
          <div className="time-frame-selector">
            <button onClick={() => handleSearch('1month')}>1M</button>
            <button onClick={() => handleSearch('3months')}>3M</button>
            <button onClick={() => handleSearch('1year')}>1Y</button>
            <button onClick={() => handleSearch('5years')}>5Y</button>
          </div>
          <div className="watchlist">
            <h3>Watchlist</h3>
            {/* We'll implement this later */}
          </div>
        </aside>
        <main className="main-content">
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
              <div className="analysis-container">
                <TechnicalAnalysis technicalIndicators={technicalIndicators} />
                <VolumeAnalysis historicalData={historicalData} />
              </div>
              <div className="ml-predictions">
                <h3>ML Predictions</h3>
                <p>Coming soon: Advanced stock predictions using machine learning models.</p>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default Dashboard;
