
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ChartLine, Home, TrendingUp, FileText, BarChart } from "lucide-react";

const Header = () => {
  const location = useLocation();
  
  const isActive = (path: string) => location.pathname === path;

  return (
    <header className="w-full bg-card/80 backdrop-blur-md py-4 px-6 md:px-12 border-b border-border sticky top-0 z-50">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-4">
          <Link to="/" className="flex items-center gap-2 group">
            <div className="bg-primary rounded-full p-2 group-hover:animate-pulse transition-all">
              <TrendingUp className="h-6 w-6 text-white" />
            </div>
            <span className="text-xl font-bold gradient-text">MarketVision</span>
          </Link>

          <nav className="flex flex-wrap gap-1 md:gap-2">
            <Link 
              to="/" 
              className={`flex items-center gap-2 px-3 py-2 rounded-md hover:bg-secondary transition-all ${isActive('/') ? 'bg-secondary neon-glow text-primary' : 'text-foreground'}`}
            >
              <Home className={`h-4 w-4 ${isActive('/') ? 'text-primary' : ''}`} />
              <span>Home</span>
            </Link>
            <Link 
              to="/sentiment" 
              className={`flex items-center gap-2 px-3 py-2 rounded-md hover:bg-secondary transition-all ${isActive('/sentiment') ? 'bg-secondary accent-glow text-accent' : 'text-foreground'}`}
            >
              <BarChart className={`h-4 w-4 ${isActive('/sentiment') ? 'text-accent' : ''}`} />
              <span>Sentiment</span>
            </Link>
            <Link 
              to="/predict" 
              className={`flex items-center gap-2 px-3 py-2 rounded-md hover:bg-secondary transition-all ${isActive('/predict') ? 'bg-secondary success-glow text-success' : 'text-foreground'}`}
            >
              <ChartLine className={`h-4 w-4 ${isActive('/predict') ? 'text-success' : ''}`} />
              <span>Predict</span>
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;
