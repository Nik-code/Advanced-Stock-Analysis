import axios from 'axios';

const API_URL = 'http://localhost:8000'; // Replace with your actual API URL

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
  try {
    const response = await api.get('/');
    return response.data;
  } catch (error) {
    console.error('Error checking API status:', error);
    throw error;
  }
};

// Get Quote
export const getQuote = async (instruments: string) => {
  try {
    const response = await api.get(`/api/quote?instruments=${instruments}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching quote:', error);
    throw error;
  }
};

// Historical Data
export const getHistoricalData = async (code: string, days: number = 365) => {
  try {
    const response = await api.get(`/api/historical/${code}?days=${days}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching historical data:', error);
    throw error;
  }
};

// Login
export const getLoginURL = async () => {
  try {
    const response = await api.get('/api/login');
    return response.data;
  } catch (error) {
    console.error('Error getting login URL:', error);
    throw error;
  }
};

// Callback
export const handleCallback = async (requestToken: string) => {
  try {
    const response = await api.get(`/api/callback?request_token=${requestToken}`);
    return response.data;
  } catch (error) {
    console.error('Error handling callback:', error);
    throw error;
  }
};

// Technical Indicators
export const getTechnicalIndicators = async (symbol: string, days: number = 365) => {
  try {
    const response = await api.get(`/api/stocks/${symbol}/indicators?days=${days}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching technical indicators:', error);
    throw error;
  }
};

// Real-time Data
export const getRealtimeData = async (symbol: string) => {
  try {
    const response = await api.get(`/api/stocks/${symbol}/realtime`);
    return response.data;
  } catch (error) {
    console.error('Error fetching real-time data:', error);
    throw error;
  }
};

// Market Overview
export const getMarketOverview = async (limit: number = 10) => {
  try {
    const response = await api.get(`/api/market/overview?limit=${limit}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching market overview:', error);
    throw error;
  }
};

// Compare Stocks
export const compareStocks = async (symbols: string[], days: number = 365) => {
  try {
    const symbolsString = symbols.join(',');
    const response = await api.get(`/api/stocks/compare?symbols=${symbolsString}&days=${days}`);
    return response.data;
  } catch (error) {
    console.error('Error comparing stocks:', error);
    throw error;
  }
};

export default api;