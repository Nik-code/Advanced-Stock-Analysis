import React from 'react';
import './Header.css';

const Header: React.FC = () => {
  return (
    <header className={styles.header}>
      <input type="text" placeholder="Search for various stocks" className={styles.searchBar} />
      <div className={styles.userInfo}>
        {/* We'll add user info and icons here later */}
        User Info
      </div>
    </header>
  );
};

export default Header;