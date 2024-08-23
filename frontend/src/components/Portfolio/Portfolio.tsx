import React from 'react';
import './Portfolio.css';

const Portfolio: React.FC = () => {
  return (
    <div className="portfolio">
      <h2>My Portfolio</h2>
      <div className="stock-list">
        {/* Add stock items here */}
      </div>
      <div className="stock-chart">
        {/* Add stock chart here */}
      </div>
      <div className="watchlist">
        <h3>My watchlist</h3>
        {/* Add watchlist items here */}
      </div>
    </div>
  );
};

export default Portfolio;