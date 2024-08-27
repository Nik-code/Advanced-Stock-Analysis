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
                    onClick={() => handleSearch(tf)}
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
                    type='line'
                    data={chartData}
                    options={chartOptions}
                  />
                </Box>
                <Grid templateColumns="1fr 1fr" gap={6}>
                  <GridItem>
                    <TechnicalAnalysis technicalIndicators={technicalIndicators} />
                  </GridItem>
                  <GridItem>
                    <VolumeAnalysis historicalData={historicalData} />
                  </GridItem>
                </Grid>
                <Box>
                  <Heading size="md" mb={2}>ML Predictions</Heading>
                  <Text>Coming soon: Advanced stock predictions using machine learning models.</Text>
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
