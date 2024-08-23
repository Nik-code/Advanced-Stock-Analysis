import React from 'react';
import './Header.css';

const Header: React.FC = () => {
  return (
    <header className="header">
      <input type="text" placeholder="Search for various stocks" className="searchBar" />
      <div className="userInfo">
        {/* We'll add user info and icons here later */}
        User Info
      </div>
    </header>
  );
};

export default Header;