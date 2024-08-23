import React from 'react';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard/Dashboard';

const App: React.FC = () => {
  return (
    <Layout>
      <Dashboard />
    </Layout>
  );
};

export default App;