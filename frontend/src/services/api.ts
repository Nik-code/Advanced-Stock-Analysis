import axios from 'axios';

const API_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
    'X-Kite-Version': '3',
  },
});

// Helper function to set the access token
export const setAccessToken = (token: string) => {
  api.defaults.headers.common['Authorization'] = `token ${token}`;
};

// Root endpoint
export const checkAPIStatus = async () => {
  const response = await api.get('/');
  return response.data;
};

// Get Quote
export const getQuote = async (instruments: string) => {
  const response = await api.get(`/api/quote?instruments=${instruments}`);
  return response.data;
};

// Historical Data
export const getHistoricalData = async (code: string, timeFrame: string = '1year') => {
  const response = await api.get(`/api/historical/${code}`, { params: { timeFrame } });
  return response.data;
};

// Login
export const getLoginURL = async () => {
  const response = await api.get('/api/login');
  return response.data;
};

// Callback
export const handleCallback = async (requestToken: string) => {
  const response = await api.get(`/api/callback?request_token=${requestToken}`);
  return response.data;
};

// Technical Indicators
export const getTechnicalIndicators = async (code: string, timeFrame: string = '1year') => {
  const response = await api.get(`api/stocks/${code}/indicators`, { params: { timeFrame } });
  return response.data;
};

// Real-time Data
export const getRealtimeData = async (symbol: string) => {
  const response = await api.get(`/api/stocks/${symbol}/realtime`);
  return response.data;
};

// Market Overview
export const getMarketOverview = async (limit: number = 10) => {
  const response = await api.get(`/api/market/overview?limit=${limit}`);
  return response.data;
};

// Compare Stocks
export const compareStocks = async (symbols: string[], days: number = 365) => {
  const symbolsString = symbols.join(',');
  const response = await api.get(`/api/stocks/compare?symbols=${symbolsString}&days=${days}`);
  return response.data;
};

export const getMarketSentiment = async (symbol: string) => {
  const response = await api.get(`/api/stocks/${symbol}/sentiment`);
  return response.data;
};

export const getNewsArticles = async (symbol: string) => {
  const response = await api.get(`/api/stocks/${symbol}/news`);
  return response.data;
};

export const getPeerComparison = async (symbol: string) => {
  const response = await api.get(`/api/stocks/${symbol}/peers`);
  return response.data;
};

export const getPrediction = async (stockCode: string, historicalData: number[]) => {
  const response = await api.post(`/api/predict/${stockCode}`, { data: historicalData });
  return response.data;
};

export const getSentiment = async (symbol: string) => {
  const response = await api.get(`/api/sentiment/${symbol}`);
  return response.data;
};