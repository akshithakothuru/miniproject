
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { ChartLine, ArrowRight, TrendingUp, BarChart, CheckCircle, FileCog } from "lucide-react";
import Header from '../components/Header';
import Footer from '../components/Footer';
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const Index = () => {
  const [showAnimation, setShowAnimation] = useState(false);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setShowAnimation(true);
    }, 300);
    
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      
      <main className="flex-grow">
        {/* Hero Section */}
        <section className="py-16 md:py-24">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col md:flex-row items-center gap-8 md:gap-16">
              <div className="flex-1 text-center md:text-left space-y-6 animate-slide-down">
                <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold leading-tight">
                  Market<span className="gradient-text">Vision</span> Insights Hub
                </h1>
                <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl">
                  Predicting Tomorrow's Stock Prices Today
                </p>
                <p className="text-lg text-muted-foreground max-w-2xl">
                  Advanced stock forecasting using sentiment analysis LSTM deep learning networks.
                </p>
                <div className="flex flex-col sm:flex-row gap-4 justify-center md:justify-start">
                  <Button asChild size="lg" className="bg-gradient-to-r from-primary to-accent hover:brightness-110 transition-all">
                    <Link to="/sentiment" className="flex items-center gap-2">
                      Get Started
                      <ArrowRight className="h-4 w-4" />
                    </Link>
                  </Button>
                </div>
              </div>
              
              <div className="flex-1 relative animate-slide-up">
                <div className="relative w-full max-w-md mx-auto">
                  {/* Animated glowing background */}
                  <div className="absolute inset-0 bg-primary/10 rounded-xl animate-pulse"></div>
                  
                  <div className="relative glass-card rounded-xl p-6">
                    <div className="flex justify-between items-center mb-6">
                      <div>
                        <h3 className="font-bold text-primary">TSLA</h3>
                        <p className="text-sm text-muted-foreground">Tesla Inc.</p>
                      </div>
                      <div className="text-right">
                        <p className="text-success font-bold">$278.14</p>
                        <p className="text-sm text-success">+2.35%</p>
                      </div>
                    </div>
                    
                    {/* Stylized Chart */}
                    <div className="h-32 w-full relative mb-4">
                      <svg className="w-full h-full" viewBox="0 0 100 40">
                        <path
                          d="M0,30 L10,28 L20,32 L30,25 L40,22 L50,18 L60,20 L70,15 L80,10 L90,8 L100,5"
                          fill="none"
                          stroke="#0ea5e9"
                          strokeWidth="2"
                          strokeDasharray="200"
                          strokeDashoffset={showAnimation ? "0" : "200"}
                          style={{ transition: "stroke-dashoffset 2s ease-out" }}
                        />
                        <path
                          d="M0,30 L10,28 L20,32 L30,25 L40,22 L50,18 L60,20 L70,15 L80,10 L90,8 L100,5 V40 H0 Z"
                          fill="url(#gradient)"
                          opacity={showAnimation ? "0.2" : "0"}
                          style={{ transition: "opacity 2s ease-out" }}
                        />
                        <defs>
                          <linearGradient id="gradient" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" stopColor="#0ea5e9" />
                            <stop offset="100%" stopColor="#0c4a6e" stopOpacity="0" />
                          </linearGradient>
                        </defs>
                      </svg>
                      
                      {/* Prediction Point */}
                      <div className="absolute right-0 top-[5px] h-4 w-4 bg-success rounded-full animate-pulse" />
                    </div>
                    
                    <div className="text-center text-xs text-muted-foreground">
                      <span className="text-success">▲</span> Predicted growth: +3.6% in next 5 days
                    </div>
                    
                    <div className="mt-4 pt-4 border-t border-border">
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Sentiment Score:</span>
                        <span className="text-success font-medium">78% Positive</span>
                      </div>
                      <div className="h-1 bg-muted rounded-full overflow-hidden mt-1">
                        <div 
                          className="h-full bg-success transition-all duration-1500"
                          style={{ width: showAnimation ? '78%' : '0%' }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* How It Works Section */}
        <section className="py-16 bg-secondary/30 border-y border-border">
          <div className="container px-4 md:px-6">
            <h2 className="text-2xl md:text-3xl font-bold text-center mb-12 gradient-text">How Our Hybrid Model Works</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl mx-auto">
              <div className="relative">
                <div className="glass-card p-6 animate-float hover-lift">
                  <div className="h-12 w-12 rounded-full bg-accent/20 flex items-center justify-center mb-4">
                    <BarChart className="h-6 w-6 text-accent" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">1. Sentiment Analysis</h3>
                  <p className="text-muted-foreground">
                    We analyze news, social media, and market sentiment using NLP to gauge market psychology.
                  </p>
                </div>
                <div className="hidden md:block absolute top-1/2 -right-4 transform -translate-y-1/2 z-10">
                  <ArrowRight className="h-8 w-8 text-primary/30" />
                </div>
              </div>
              
              <div className="relative">
                <div className="glass-card p-6 animate-float hover-lift">
                  <div className="h-12 w-12 rounded-full bg-primary/20 flex items-center justify-center mb-4">
                    <FileCog className="h-6 w-6 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">2. Hybrid Modeling</h3>
                  <p className="text-muted-foreground">
                    Our systems combine traditional  models with deep learning LSTM networks.
                  </p>
                </div>
                <div className="hidden md:block absolute top-1/2 -right-4 transform -translate-y-1/2 z-10">
                  <ArrowRight className="h-8 w-8 text-primary/30" />
                </div>
              </div>
              
              <div className="relative">
                <div className="glass-card p-6 animate-float hover-lift">
                  <div className="h-12 w-12 rounded-full bg-success/20 flex items-center justify-center mb-4">
                    <ChartLine className="h-6 w-6 text-success" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">3. Price Prediction</h3>
                  <p className="text-muted-foreground">
                    The integrated model generates 5-day forecasts with high accuracy R² scores.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-16">
          <div className="container px-4 md:px-6">
            <h2 className="text-3xl font-bold text-center mb-12 gradient-text">Advanced Stock Analysis Features</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <Card className="p-6 hover-lift glass-card animate-slide-up" style={{ animationDelay: "0.1s" }}>
                <div className="h-12 w-12 rounded-full bg-accent/20 flex items-center justify-center mb-4">
                  <BarChart className="h-6 w-6 text-accent" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Sentiment Analysis</h3>
                <p className="text-muted-foreground">
                  Real-time analysis of news and social media sentiment to gauge market psychology and investor emotions.
                </p>
              </Card>
              
              <Card className="p-6 hover-lift glass-card animate-slide-up" style={{ animationDelay: "0.2s" }}>
                <div className="h-12 w-12 rounded-full bg-primary/20 flex items-center justify-center mb-4">
                  <ChartLine className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Price Prediction</h3>
                <p className="text-muted-foreground">
                  Hybrid LSTM model predicts stock prices for the next 5 trading days with high accuracy.
                </p>
              </Card>
              
              <Card className="p-6 hover-lift glass-card animate-slide-up" style={{ animationDelay: "0.3s" }}>
                <div className="h-12 w-12 rounded-full bg-success/20 flex items-center justify-center mb-4">
                  <CheckCircle className="h-6 w-6 text-success" />
                </div>
                <h3 className="text-xl font-semibold mb-2">Performance Metrics</h3>
                <p className="text-muted-foreground">
                  View R² score and model accuracy to understand prediction confidence and reliability.
                </p>
              </Card>
            </div>
          </div>
        </section>

        {/* Call to Action */}
        <section className="py-16 bg-secondary/30 border-y border-border">
          <div className="container px-4 md:px-6">
            <div className="text-center max-w-3xl mx-auto">
              <h2 className="text-2xl md:text-3xl font-bold mb-4 gradient-text">
                Ready to make data-driven investment decisions?
              </h2>
              <p className="text-lg text-muted-foreground mb-8">
                Start by analyzing market sentiment and predicting stock prices with our advanced AI tools.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button asChild size="lg" className="bg-gradient-to-r from-primary to-accent hover:brightness-110 transition-all">
                  <Link to="/sentiment" className="flex items-center gap-2">
                    <BarChart className="h-5 w-5" />
                    Analyze Sentiment
                  </Link>
                </Button>
                <Button asChild size="lg" variant="outline" className="border-primary/30 hover:border-primary">
                  <Link to="/predict" className="flex items-center gap-2">
                    <ChartLine className="h-5 w-5" />
                    Predict Prices
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default Index;
