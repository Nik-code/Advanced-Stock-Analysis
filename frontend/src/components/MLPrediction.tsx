import React, { useState } from 'react';
import { Box, VStack, Text, Button, Spinner } from '@chakra-ui/react';
import axios from 'axios';

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
