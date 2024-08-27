import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  ChartOptions,
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import { enUS } from 'date-fns/locale';
import './TechnicalAnalysis.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

interface TechnicalAnalysisProps {
  technicalIndicators: any;
}

const TechnicalAnalysis: React.FC<TechnicalAnalysisProps> = ({ technicalIndicators }) => {
  const chartData = {
    labels: technicalIndicators?.indicators?.SMA_20.map((item: any) => item.date),
    datasets: [
      {
        label: 'SMA 20',
        data: technicalIndicators?.indicators?.SMA_20.map((item: any) => item.value),
        borderColor: 'rgb(75, 192, 192)',
        fill: false,
      },
      {
        label: 'EMA 20',
        data: technicalIndicators?.indicators?.EMA_20.map((item: any) => item.value),
        borderColor: 'rgb(255, 99, 132)',
        fill: false,
      },
      {
        label: 'RSI',
        data: technicalIndicators?.indicators?.RSI.map((item: any) => item.value),
        borderColor: 'rgb(54, 162, 235)',
        fill: false,
      },
    ],
  };

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    scales: {
      x: {
        type: 'time' as const,
        time: {
          unit: 'day',
        },
      },
      y: {
        beginAtZero: false,
      },
    },
  };

  return (
    <div className="technical-analysis">
      <h3>Technical Indicators</h3>
      <Line data={chartData} options={chartOptions} />
      <div className="indicator-values">
        <p>SMA 20: {technicalIndicators?.indicators?.SMA_20[technicalIndicators.indicators.SMA_20.length - 1]?.value?.toFixed(2)}</p>
        <p>EMA 20: {technicalIndicators?.indicators?.EMA_20[technicalIndicators.indicators.EMA_20.length - 1]?.value?.toFixed(2)}</p>
        <p>RSI: {technicalIndicators?.indicators?.RSI[technicalIndicators.indicators.RSI.length - 1]?.value?.toFixed(2)}</p>
        <p>MACD: {technicalIndicators?.indicators?.MACD?.macd_line[technicalIndicators.indicators.MACD.macd_line.length - 1]?.value?.toFixed(2)}</p>
      </div>
    </div>
  );
};

export default TechnicalAnalysis;