import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import './App.css'; // Change this from module to regular CSS

const App: React.FC = () => {
  return (
    <Router>
      <div className="app">
        <Layout>
          <Dashboard />
        </Layout>
      </div>
    </Router>
  );
};

export default App;