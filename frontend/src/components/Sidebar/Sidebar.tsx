import React from 'react';
import { Box, VStack, Icon, Tooltip } from '@chakra-ui/react';
import { Link } from 'react-router-dom';
import { FiHome, FiTrendingUp, FiStar, FiFileText, FiSettings } from 'react-icons/fi';

interface SidebarItemProps {
  icon: React.ElementType;
  label: string;
  to: string;
}

const SidebarItem: React.FC<SidebarItemProps> = ({ icon, label, to }) => (
  <Tooltip label={label} placement="right">
    <Box as={Link} to={to} p={3} borderRadius="md" _hover={{ bg: 'whiteAlpha.200' }}>
      <Icon as={icon} boxSize={6} />
    </Box>
  </Tooltip>
);

const Sidebar: React.FC = () => {
  return (
    <Box w="60px" bg="gray.900" color="white" py={4}>
      <VStack spacing={4}>
        <SidebarItem icon={FiHome} label="Dashboard" to="/" />
        <SidebarItem icon={FiTrendingUp} label="Analysis" to="/analysis" />
        <SidebarItem icon={FiStar} label="Watchlist" to="/watchlist" />
        <SidebarItem icon={FiFileText} label="News" to="/news" />
        <SidebarItem icon={FiSettings} label="Settings" to="/settings" />
      </VStack>
    </Box>
  );
};

export default Sidebar;