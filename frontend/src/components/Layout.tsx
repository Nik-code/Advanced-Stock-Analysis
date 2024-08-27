import React from 'react';
import { Box, Flex } from '@chakra-ui/react';
import Sidebar from './Sidebar/Sidebar';
import Header from './Header/Header';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <Flex h="100vh">
      <Sidebar />
      <Flex flexDirection="column" flex={1}>
        <Header />
        <Box as="main" flex={1} p={4} overflowY="auto">
          {children}
        </Box>
      </Flex>
    </Flex>
  );
};

export default Layout;