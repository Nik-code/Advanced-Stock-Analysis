import React from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, TimeScale } from 'chart.js';
import 'chartjs-adapter-date-fns';
import './VolumeAnalysis.css';
import { enUS } from 'date-fns/locale';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, TimeScale);

interface VolumeAnalysisProps {
  historicalData: any[];
}

const VolumeAnalysis: React.FC<VolumeAnalysisProps> = ({ historicalData }) => {
  const chartData = {
    labels: historicalData.map(data => data.date),
    datasets: [
      {
        label: 'Volume',
        data: historicalData.map(data => data.volume),
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'day',
        },
        adapters: {
          date: {
            locale: enUS,
          },
        },
      },
      y: {
        beginAtZero: true,
      },
    },
  };

  const averageVolume = historicalData.reduce((sum, data) => sum + data.volume, 0) / historicalData.length;
  const maxVolume = Math.max(...historicalData.map(data => data.volume));
  const minVolume = Math.min(...historicalData.map(data => data.volume));

  return (
    <div className="volume-analysis">
      <h3>Volume Analysis</h3>
      <Bar data={chartData} options={chartOptions} />
      <div className="volume-stats">
        <p>Average Volume: {averageVolume.toFixed(0)}</p>
        <p>Max Volume: {maxVolume.toFixed(0)}</p>
        <p>Min Volume: {minVolume.toFixed(0)}</p>
      </div>
    </div>
  );
};

export default VolumeAnalysis;