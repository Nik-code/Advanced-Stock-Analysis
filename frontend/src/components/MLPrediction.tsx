import React, { useState, useEffect } from 'react';
import { Box, VStack, Text, Button, Spinner } from '@chakra-ui/react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

interface MLPredictionProps {
  stockCode: string;
  historicalData: any[];
}

const MLPrediction: React.FC<MLPredictionProps> = ({ stockCode, historicalData }) => {
  const [predictions, setPredictions] = useState<number[]>([]);
  const [pastPredictions, setPastPredictions] = useState<number[]>([]);
  const [sentiment, setSentiment] = useState<number | null>(null);
  const [explanation, setExplanation] = useState<string>('');
  const [analysis, setAnalysis] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePrediction = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [predictionResponse, sentimentResponse] = await Promise.all([
        axios.post(`http://localhost:8000/api/predict/${stockCode}`, historicalData.map(d => d.close)),
        axios.get(`http://localhost:8000/api/sentiment/${stockCode}`)
      ]);
      setPredictions(predictionResponse.data.predictions);
      setPastPredictions(predictionResponse.data.past_predictions);
      setSentiment(sentimentResponse.data.sentiment);
      setExplanation(sentimentResponse.data.explanation);
      setAnalysis(sentimentResponse.data.analysis);
    } catch (error) {
      console.error('Error fetching predictions or sentiment:', error);
      setError('Failed to fetch predictions or sentiment. Please try again.');
    }
    setIsLoading(false);
  };

  const chartData = {
    labels: [...historicalData.slice(-60).map(d => d.date), ...Array(7).fill('').map((_, i) => `Day ${i + 1}`)],
    datasets: [
      {
        label: 'Historical Close Price',
        data: historicalData.slice(-60).map(d => d.close),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      },
      {
        label: 'Past Predictions',
        data: pastPredictions.length > 0 ? [...Array(60 - pastPredictions.length).fill(null), ...pastPredictions] : [],
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1
      },
      {
        label: 'Future Predictions',
        data: predictions.length > 0 ? [...Array(60).fill(null), ...predictions] : [],
        borderColor: 'rgb(255, 159, 64)',
        tension: 0.1
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Stock Price Prediction',
      },
    },
  };

  return (
    <Box>
      <Button onClick={handlePrediction} isLoading={isLoading}>
        Generate Prediction and Analysis
      </Button>
      {error && <Text color="red.500">{error}</Text>}
      {sentiment !== null && (
        <VStack align="start" spacing={4} mt={4}>
          <Text fontWeight="bold">Sentiment Score: {sentiment.toFixed(2)}</Text>
          <Text fontWeight="bold">Explanation: {explanation}</Text>
          <Text fontWeight="bold">Analysis:</Text>
          <Text>{analysis}</Text>
          <Text fontWeight="bold">LSTM Prediction Chart:</Text>
          {historicalData.length > 0 && (
            <Box width="100%" height="400px">
              <Line data={chartData} options={chartOptions} />
            </Box>
          )}
          <Text fontWeight="bold">LSTM Prediction (Next 7 days):</Text>
          {predictions.map((pred, index) => (
            <Text key={index}>Day {index + 1}: {pred.toFixed(2)}</Text>
          ))}
        </VStack>
      )}
    </Box>
  );
};

export default MLPrediction;
