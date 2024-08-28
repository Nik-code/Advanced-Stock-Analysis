// TechnicalAnalysis.tsx

import React from 'react';
import { Line } from 'react-chartjs-2';
import { ChartOptions } from 'chart.js';
import './TechnicalAnalysis.css';

interface TechnicalIndicatorsData {
  sma_20: number[];
  ema_50: number[];
  rsi_14: (number | null)[];
  macd: number[];
  macd_signal: number[];
  bollinger_upper: (number | null)[];
  bollinger_middle: number[];
  bollinger_lower: (number | null)[];
  atr: number[];
  dates: string[];
}

interface TechnicalAnalysisProps {
  technicalIndicators: TechnicalIndicatorsData;
  timeFrame: string;
}

const TechnicalAnalysis: React.FC<TechnicalAnalysisProps> = ({ technicalIndicators, timeFrame }) => {
  if (!technicalIndicators) {
    return <p>No technical indicators available for the selected time frame.</p>;
  }

  const chartData = {
    labels: technicalIndicators.dates,
    datasets: [
      {
        label: 'SMA 20',
        data: technicalIndicators.sma_20,
        borderColor: 'rgb(75, 192, 192)',
        fill: false,
      },
      {
        label: 'EMA 50',
        data: technicalIndicators.ema_50,
        borderColor: 'rgb(255, 99, 132)',
        fill: false,
      },
      {
        label: 'RSI',
        data: technicalIndicators.rsi_14,
        borderColor: 'rgb(54, 162, 235)',
        fill: false,
        yAxisID: 'rsi',
      },
    ],
  };

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    scales: {
      x: {
        type: 'time',
        time: {
          unit: timeFrame === '1month' ? 'day' : timeFrame === '3months' ? 'week' : 'month',
        },
      },
      y: {
        beginAtZero: false,
        title: {
          display: true,
          text: 'Price',
        },
      },
      rsi: {
        position: 'right',
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: 'RSI',
        },
      },
    },
  };

  return (
    <div className="technical-analysis">
      <h3>Technical Indicators</h3>
      <Line data={chartData} options={chartOptions} />
      <div className="indicator-values">
        <p>SMA 20: {technicalIndicators.sma_20[technicalIndicators.sma_20.length - 1]?.toFixed(2) || 'N/A'}</p>
        <p>EMA 50: {technicalIndicators.ema_50[technicalIndicators.ema_50.length - 1]?.toFixed(2) || 'N/A'}</p>
        <p>RSI: {technicalIndicators.rsi_14[technicalIndicators.rsi_14.length - 1]?.toFixed(2) || 'N/A'}</p>
        <p>MACD: {technicalIndicators.macd[technicalIndicators.macd.length - 1]?.toFixed(2) || 'N/A'}</p>
        <p>ATR: {technicalIndicators.atr[technicalIndicators.atr.length - 1]?.toFixed(2) || 'N/A'}</p>
      </div>
    </div>
  );
};

export default TechnicalAnalysis;