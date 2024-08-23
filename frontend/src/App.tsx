import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard/Dashboard';
import Wallet from './pages/Wallet/Wallet';
import News from './pages/News/News';
import StockFund from './pages/StockFund/StockFund';
import Community from './pages/Community/Community';
import Settings from './pages/Settings/Settings';
import Contact from './pages/Contact/Contact';

const App: React.FC = () => {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/wallet" element={<Wallet />} />
          <Route path="/news" element={<News />} />
          <Route path="/stock-fund" element={<StockFund />} />
          <Route path="/community" element={<Community />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/contact" element={<Contact />} />
        </Routes>
      </Layout>
    </Router>
  );
};

export default App;