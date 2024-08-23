export const getStocks = async () => {
  // This is a mock implementation. Replace with actual API call later.
  return [
    { symbol: 'AAPL', name: 'Apple Inc.', price: 150.25, change: 2.5 },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 2750.80, change: -1.2 },
    { symbol: 'MSFT', name: 'Microsoft Corporation', price: 305.60, change: 0.8 },
  ];
};