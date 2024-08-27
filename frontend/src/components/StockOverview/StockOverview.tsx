import React from 'react';
import { Box, Heading, Text, SimpleGrid, Stat, StatLabel, StatNumber, StatHelpText, StatArrow } from '@chakra-ui/react';

interface StockOverviewProps {
  stockCode: string;
  quoteData: any;
}

const StockOverview: React.FC<StockOverviewProps> = ({ stockCode, quoteData }) => {
  return (
    <Box>
      <Heading size="lg" mb={2}>{stockCode} - {quoteData?.last_price}</Heading>
      <Stat>
        <StatLabel>Change</StatLabel>
        <StatNumber>{quoteData?.change}</StatNumber>
        <StatHelpText>
          <StatArrow type={quoteData?.change_percent >= 0 ? 'increase' : 'decrease'} />
          {quoteData?.change_percent}%
        </StatHelpText>
      </Stat>
      <SimpleGrid columns={2} spacing={4} mt={4}>
        <Stat>
          <StatLabel>Open</StatLabel>
          <StatNumber>{quoteData?.open}</StatNumber>
        </Stat>
        <Stat>
          <StatLabel>High</StatLabel>
          <StatNumber>{quoteData?.high}</StatNumber>
        </Stat>
        <Stat>
          <StatLabel>Low</StatLabel>
          <StatNumber>{quoteData?.low}</StatNumber>
        </Stat>
        <Stat>
          <StatLabel>Volume</StatLabel>
          <StatNumber>{quoteData?.volume}</StatNumber>
        </Stat>
        <Stat>
          <StatLabel>52W High</StatLabel>
          <StatNumber>{quoteData?.year_high}</StatNumber>
        </Stat>
        <Stat>
          <StatLabel>52W Low</StatLabel>
          <StatNumber>{quoteData?.year_low}</StatNumber>
        </Stat>
      </SimpleGrid>
    </Box>
  );
};

export default StockOverview;