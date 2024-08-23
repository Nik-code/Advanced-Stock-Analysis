import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import StockList from './components/StockList';

const App: React.FC = () => {
  return (
    <Router>
      <div className="App">
        <h1>BSE Trading Dashboard</h1>
        <Switch>
          <Route exact path="/" component={StockList} />
          {/* Add more routes here as you develop other components */}
        </Switch>
      </div>
    </Router>
  );
};

export default App;