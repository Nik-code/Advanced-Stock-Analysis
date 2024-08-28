import React from 'react';
import { Box, Grid, Heading, Text, VStack, HStack, Stat, StatLabel, StatNumber, StatHelpText, StatArrow } from '@chakra-ui/react';

const Dashboard: React.FC = () => {
  // Mock data - replace with actual API calls later
  const topPerformers = [
    { symbol: 'AAPL', name: 'Apple Inc.', change: 2.5 },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', change: 1.8 },
    { symbol: 'MSFT', name: 'Microsoft Corporation', change: 1.2 },
  ];

  const needAttention = [
    { symbol: 'TSLA', name: 'Tesla, Inc.', change: -3.2 },
    { symbol: 'FB', name: 'Facebook, Inc.', change: -2.1 },
    { symbol: 'NFLX', name: 'Netflix, Inc.', change: -1.5 },
  ];

  return (
    <Box>
      <Heading mb={6}>Dashboard</Heading>
      <Grid templateColumns="repeat(2, 1fr)" gap={6}>
        <Box>
          <Heading size="md" mb={4}>Top Performers</Heading>
          <VStack align="stretch" spacing={4}>
            {topPerformers.map((stock) => (
              <HStack key={stock.symbol} justify="space-between" p={4} bg="whiteAlpha.100" borderRadius="md">
                <VStack align="start" spacing={0}>
                  <Text fontWeight="bold">{stock.symbol}</Text>
                  <Text fontSize="sm" color="gray.500">{stock.name}</Text>
                </VStack>
                <Stat>
                  <StatNumber color="green.400">+{stock.change}%</StatNumber>
                </Stat>
              </HStack>
            ))}
          </VStack>
        </Box>
        <Box>
          <Heading size="md" mb={4}>Need Attention</Heading>
          <VStack align="stretch" spacing={4}>
            {needAttention.map((stock) => (
              <HStack key={stock.symbol} justify="space-between" p={4} bg="whiteAlpha.100" borderRadius="md">
                <VStack align="start" spacing={0}>
                  <Text fontWeight="bold">{stock.symbol}</Text>
                  <Text fontSize="sm" color="gray.500">{stock.name}</Text>
                </VStack>
                <Stat>
                  <StatNumber color="red.400">{stock.change}%</StatNumber>
                </Stat>
              </HStack>
            ))}
          </VStack>
        </Box>
      </Grid>
    </Box>
  );
};

export default Dashboard;