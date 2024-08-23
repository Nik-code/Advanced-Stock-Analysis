import React from 'react';
import { Link } from 'react-router-dom';
import './Sidebar.css';

const Sidebar: React.FC = () => {
  return (
    <aside className="sidebar">
      <div className="logo">
        <span className="icon">⚡</span> GoStock
      </div>
      <div className="investment">
        <p>Total Investment</p>
        <h2>$5380,90</h2>
        <span className="percentage">+18,10% ↑</span>
      </div>
      <nav className="nav">
        <ul>
          <li><Link to="/">Home</Link></li>
          <li><Link to="/dashboard">Dashboard</Link></li>
          <li><Link to="/wallet">Wallet</Link></li>
          <li><Link to="/news">News</Link></li>
          <li><Link to="/stock-fund">Stock & Fund</Link></li>
          <li><Link to="/community">Our Community</Link></li>
          <li><Link to="/settings">Settings</Link></li>
          <li><Link to="/contact">Contact us</Link></li>
        </ul>
      </nav>
    </aside>
  );
};

export default Sidebar;