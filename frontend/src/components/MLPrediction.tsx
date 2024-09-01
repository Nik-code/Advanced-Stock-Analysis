import React, { useState } from 'react';
import axios from 'axios';
import { Button, Box, Text } from '@chakra-ui/react';
import { Line } from 'react-chartjs-2';

interface MLPredictionProps {
  stockCode: string;
  historicalData: any[];
}

const MLPrediction: React.FC<MLPredictionProps> = ({ stockCode, historicalData }) => {
  const [predictions, setPredictions] = useState<number[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePrediction = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.post(`http://localhost:8000/api/predict/${stockCode}`, historicalData.map(d => d.close));
      setPredictions(response.data.predictions);
    } catch (error) {
      console.error('Error fetching predictions:', error);
      setError('Failed to fetch predictions. Please try again.');
    }
    setIsLoading(false);
  };

  const chartData = {
    labels: [...historicalData.map(d => d.date), ...Array(7).fill('').map((_, i) => `Day ${i + 1}`)],
    datasets: [
      {
        label: 'Historical Close Price',
        data: historicalData.map(d => d.close),
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      },
      {
        label: 'Predicted Close Price',
        data: [...Array(historicalData.length).fill(null), ...predictions],
        fill: false,
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1
      }
    ]
  };

  return (
    <Box>
      <Button onClick={handlePrediction} isLoading={isLoading} mb={4}>
        Show ML Prediction
      </Button>
      {error && <Text color="red.500" mb={4}>{error}</Text>}
      {predictions.length > 0 && (
        <Line data={chartData} options={{ responsive: true }} />
      )}
    </Box>
  );
};

export default MLPrediction;
