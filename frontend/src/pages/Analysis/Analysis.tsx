import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  VStack,
  HStack,
  Heading,
  Text,
  Input,
  Button,
  useColorMode,
  Grid,
  GridItem,
} from '@chakra-ui/react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, TimeScale, ChartData, ChartOptions } from 'chart.js';
import { Chart as ReactChart } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import { getHistoricalData, getTechnicalIndicators, getQuote } from '../../services/api';
import StockOverview from '../../components/StockOverview/StockOverview';
import TechnicalAnalysis from '../../components/TechnicalAnalysis/TechnicalAnalysis';
import VolumeAnalysis from '../../components/VolumeAnalysis/VolumeAnalysis';
import axios from 'axios';
import PredictionGraph from '../components/PredictionGraph';

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

const Analysis: React.FC = () => {
  const { colorMode, toggleColorMode } = useColorMode();
  const [stockCode, setStockCode] = useState<string>('');
  const [timeFrame, setTimeFrame] = useState<string>('1year');
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [technicalIndicators, setTechnicalIndicators] = useState<any>(null);
  const [quoteData, setQuoteData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);

  const chartRef = useRef<ChartJS<'line' | 'bar'> | null>(null);

  const chartData: ChartData<'line' | 'bar', (number | null)[], string> = {
    labels: historicalData.map(data => new Date(data.date).toISOString()),
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
      x: {
        type: 'time',
        time: {
          unit: timeFrame === '1month' ? 'day' : timeFrame === '3months' ? 'week' : 'month',
        },
      },
      y: {
        type: 'linear' as const,
        display: true,
        position: 'left' as const,
        title: {
          display: true,
          text: 'Close Price'
        }
      },
      y1: {
        type: 'linear' as const,
        display: true,
        position: 'right' as const,
        title: {
          display: true,
          text: 'Volume'
        },
        grid: {
          drawOnChartArea: false,
        },
      },
    },
  };

  const handleSearch = async (newTimeFrame?: string) => {
    console.log('Searching for stock:', stockCode);
    if (stockCode) {
      setLoading(true);
      setError(null);
      try {
        const timeFrameToUse = newTimeFrame || timeFrame;
        const [historicalResponse, indicatorsResponse, quoteResponse] = await Promise.all([
          getHistoricalData(stockCode, timeFrameToUse),
          getTechnicalIndicators(stockCode, timeFrameToUse),
          getQuote(stockCode),
        ]);

        setHistoricalData(historicalResponse);
        setTechnicalIndicators(indicatorsResponse);
        setQuoteData(quoteResponse);
        setTimeFrame(timeFrameToUse);
      } catch (error) {
        console.error('Error fetching data:', error);
        setError('Error fetching data. Please try again.');
      } finally {
        setLoading(false);
      }
    }
  };

  const handleTimeFrameChange = (newTimeFrame: string) => {
    handleSearch(newTimeFrame);
  };

  const handlePredict = async () => {
    try {
      const response = await axios.post('/api/predict', historicalData.map(data => data.close));
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Error fetching prediction:", error);
    }
  };

  return (
    <Box>
      <VStack spacing={6} align="stretch">
        <HStack justifyContent="space-between">
          <Heading>Advanced Stock Analysis</Heading>
          <Button onClick={toggleColorMode}>
            {colorMode === 'light' ? 'Dark Mode' : 'Light Mode'}
          </Button>
        </HStack>

        <Grid templateColumns="250px 1fr" gap={6}>
          <GridItem>
            <VStack spacing={4} align="stretch">
              <Input
                value={stockCode}
                onChange={(e) => setStockCode(e.target.value.toUpperCase())}
                placeholder="Enter stock code (e.g., TCS)"
              />
              <Button onClick={() => handleSearch()}>Search</Button>
              <HStack>
                {['1month', '3months', '1year', '5years'].map((tf) => (
                  <Button
                    key={tf}
                    onClick={() => handleTimeFrameChange(tf)}
                    variant={timeFrame === tf ? 'solid' : 'outline'}
                  >
                    {tf.charAt(0).toUpperCase() + tf.slice(1)}
                  </Button>
                ))}
              </HStack>
              <Box>
                <Heading size="md" mb={2}>Watchlist</Heading>
                <Text>Coming soon...</Text>
              </Box>
            </VStack>
          </GridItem>

          <GridItem>
            {stockCode && quoteData && (
              <VStack spacing={6} align="stretch">
                <StockOverview stockCode={stockCode} quoteData={quoteData} />
                <Box>
                  <Heading size="md" mb={2}>Historical Price and Volume Data</Heading>
                  <ReactChart
                    ref={chartRef}
                    type='bar'
                    data={chartData}
                    options={chartOptions}
                  />
                </Box>
                <Grid templateColumns="1fr 1fr" gap={6}>
                  <GridItem>
                    <TechnicalAnalysis technicalIndicators={technicalIndicators} timeFrame={timeFrame} />
                  </GridItem>
                  <GridItem>
                    <VolumeAnalysis historicalData={historicalData} />
                  </GridItem>
                </Grid>
                <Box>
                  <Heading size="md" mb={2}>ML Predictions</Heading>
                  <Button onClick={handlePredict}>Predict</Button>
                  {prediction && (
                    <>
                      <Text>Predicted Next Day Close: {prediction.toFixed(2)}</Text>
                      <PredictionGraph historicalData={historicalData} prediction={prediction} />
                    </>
                  )}
                </Box>
              </VStack>
            )}
          </GridItem>
        </Grid>
      </VStack>
    </Box>
  );
};

export default Analysis;
