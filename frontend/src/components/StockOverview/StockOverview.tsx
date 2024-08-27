import React from 'react';
import './StockOverview.css';

interface StockOverviewProps {
  stockCode: string;
  quoteData: any;
}

const StockOverview: React.FC<StockOverviewProps> = ({ stockCode, quoteData }) => {
  return (
    <div className="stock-overview">
      <h2>{stockCode} - {quoteData?.last_price}</h2>
      <p>Change: {quoteData?.change} ({quoteData?.change_percent}%)</p>
      <div className="additional-info">
        <div>
          <span>Open: {quoteData?.open}</span>
          <span>High: {quoteData?.high}</span>
          <span>Low: {quoteData?.low}</span>
        </div>
        <div>
          <span>Volume: {quoteData?.volume}</span>
          <span>52W High: {quoteData?.year_high}</span>
          <span>52W Low: {quoteData?.year_low}</span>
        </div>
      </div>
    </div>
  );
};

export default StockOverview;