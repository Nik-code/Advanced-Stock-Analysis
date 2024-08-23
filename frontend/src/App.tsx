import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import styles from './App.module.css';

const App: React.FC = () => {
  return (
    <Router>
      <div className={styles.app}>
        <Layout>
          <Dashboard />
        </Layout>
      </div>
    </Router>
  );
};

export default App;