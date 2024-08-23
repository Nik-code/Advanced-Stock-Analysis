import React from 'react';
import Sidebar from './Sidebar/Sidebar';
import Header from './Header/Header';
import './Layout.css';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="layout">
      <Sidebar />
      <div className="mainContent">
        <Header />
        <main className="content">{children}</main>
      </div>
    </div>
  );
};

export default Layout;