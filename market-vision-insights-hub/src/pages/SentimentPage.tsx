import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Link } from 'react-router-dom';
import { ArrowRight, BarChart, TrendingUp, ChartLine, FileText, AlertCircle, Loader2 } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";

const stockOptions = [
  { value: 'MSFT', label: 'Microsoft (MSFT)' },
  { value: 'TSLA', label: 'Tesla (TSLA)' },
  { value: 'AAPL', label: 'Apple (AAPL)' },
  { value: 'NFLX', label: 'Netflix (NFLX)' },
  { value: 'GOOGL', label: 'Google (GOOGL)' },
  { value: 'AMZN', label: 'Amazon (AMZN)' }
];

// This data will be shown initially and when API is unavailable
const fallbackSentimentData = {
  'MSFT': {
    score: 0.72,
    label: 'Positive',
    sources: {
      news: 0.76,
      social: 0.63,
      analyst: 0.81,
    },
    keywords: ['Cloud', 'Azure', 'AI', 'Growth', 'Enterprise'],
    newsHeadlines: [
      { title: 'Microsoft Cloud Revenue Surges 28%', url: 'https://finance.yahoo.com/news/microsoft-cloud-revenue-surges-28-123456789.html', sentiment_score: 0.042 },
      { title: 'Azure Gains Market Share Against AWS', url: 'https://finance.yahoo.com/news/azure-gains-market-share-against-aws-987654321.html', sentiment_score: 0.035 },
      { title: 'Microsoft Faces Antitrust Scrutiny', url: 'https://finance.yahoo.com/news/microsoft-faces-antitrust-scrutiny-456789123.html', sentiment_score: -0.018 },
    ]
  },
  'TSLA': {
    score: 0.48,
    label: 'Neutral',
    sources: {
      news: 0.42,
      social: 0.52,
      analyst: 0.47,
    },
    keywords: ['Production', 'China', 'Competition', 'Musk', 'EV'],
    newsHeadlines: [
      { title: 'Tesla Q1 Deliveries Beat Expectations', url: 'https://finance.yahoo.com/news/tesla-q1-deliveries-beat-expectations-123456789.html', sentiment_score: 0.029 },
      { title: 'New EV Competition Threatens Tesla Market Share', url: 'https://finance.yahoo.com/news/new-ev-competition-threatens-tesla-987654321.html', sentiment_score: -0.025 },
      { title: 'Tesla Cuts Prices in China Again', url: 'https://finance.yahoo.com/news/tesla-cuts-prices-china-again-456789123.html', sentiment_score: -0.031 },
    ]
  },
  'AAPL': {
    score: 0.84,
    label: 'Very Positive',
    sources: {
      news: 0.82,
      social: 0.89,
      analyst: 0.76,
    },
    keywords: ['iPhone', 'Services', 'Growth', 'Innovation', 'Loyal'],
    newsHeadlines: [
      { title: 'Apple Services Revenue Hits New Record', url: 'https://finance.yahoo.com/news/apple-services-revenue-hits-record-123456789.html', sentiment_score: 0.051 },
      { title: 'iPhone 15 Pro Demand Exceeds Expectations', url: 'https://finance.yahoo.com/news/iphone-15-pro-demand-exceeds-987654321.html', sentiment_score: 0.038 },
      { title: 'Apple AI Strategy Gains Momentum', url: 'https://finance.yahoo.com/news/apple-ai-strategy-gains-momentum-456789123.html', sentiment_score: 0.044 },
    ]
  },
  'NFLX': {
    score: 0.35,
    label: 'Slightly Negative',
    sources: {
      news: 0.31,
      social: 0.42,
      analyst: 0.32,
    },
    keywords: ['Subscribers', 'Content', 'Competition', 'Streaming', 'Ads'],
    newsHeadlines: [
      { title: 'Netflix Subscriber Growth Slows in Q2', url: 'https://finance.yahoo.com/news/netflix-subscriber-growth-slows-q2-123456789.html', sentiment_score: -0.027 },
      { title: 'Competition Intensifies in Streaming Space', url: 'https://finance.yahoo.com/news/competition-intensifies-streaming-space-987654321.html', sentiment_score: -0.033 },
      { title: 'New Netflix Original Shows Getting Mixed Reviews', url: 'https://finance.yahoo.com/news/netflix-original-shows-mixed-reviews-456789123.html', sentiment_score: 0.002 },
    ]
  },
  'GOOGL': {
    score: 0.62,
    label: 'Positive',
    sources: {
      news: 0.58,
      social: 0.64,
      analyst: 0.66,
    },
    keywords: ['Search', 'Ads', 'Cloud', 'AI', 'Antitrust'],
    newsHeadlines: [
      { title: 'Google Search Market Share Remains Strong', url: 'https://finance.yahoo.com/news/google-search-market-share-strong-123456789.html', sentiment_score: 0.036 },
      { title: 'Google Cloud Growth Accelerates', url: 'https://finance.yahoo.com/news/google-cloud-growth-accelerates-987654321.html', sentiment_score: 0.041 },
      { title: 'Antitrust Concerns Weigh on Google', url: 'https://finance.yahoo.com/news/antitrust-concerns-weigh-google-456789123.html', sentiment_score: -0.022 },
    ]
  },
  'AMZN': {
    score: 0.55,
    label: 'Neutral',
    sources: {
      news: 0.61,
      social: 0.49,
      analyst: 0.58,
    },
    keywords: ['E-commerce', 'AWS', 'Retail', 'Logistics', 'Prime'],
    newsHeadlines: [
      { title: 'Amazon AWS Revenue Growth Accelerates', url: 'https://finance.yahoo.com/news/amazon-aws-revenue-growth-accelerates-123456789.html', sentiment_score: 0.039 },
      { title: 'Retail Business Faces Margin Pressure', url: 'https://finance.yahoo.com/news/retail-business-faces-margin-pressure-987654321.html', sentiment_score: -0.019 },
      { title: 'Amazon Expands Same-Day Delivery Network', url: 'https://finance.yahoo.com/news/amazon-expands-same-day-delivery-456789123.html', sentiment_score: 0.028 },
    ]
  }
};

const SentimentPage = () => {
  const [selectedStock, setSelectedStock] = useState<string>('MSFT');
  const [animateScore, setAnimateScore] = useState(false);
  const [animateSources, setAnimateSources] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [backendUrl, setBackendUrl] = useState('http://localhost:8000');
  const [sentimentData, setSentimentData] = useState(fallbackSentimentData);
  const [showConfig, setShowConfig] = useState(false);
  const { toast } = useToast();
  
  useEffect(() => {
    // Reset animations when stock changes
    setAnimateScore(false);
    setAnimateSources(false);
    
    const scoreTimer = setTimeout(() => {
      setAnimateScore(true);
    }, 300);
    
    const sourcesTimer = setTimeout(() => {
      setAnimateSources(true);
    }, 800);
    
    return () => {
      clearTimeout(scoreTimer);
      clearTimeout(sourcesTimer);
    };
  }, [selectedStock]);

  const fetchSentimentData = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${backendUrl}/analyze-sentiment?ticker=${selectedStock}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch data: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('API Response:', data); // Debug log to verify the response

      // Scale the overall sentiment score from [-1, 1] to [0, 1]
      const scaledScore = (data.sentiment_score + 1) / 2; // Maps -1 -> 0, 0 -> 0.5, 1 -> 1

      // Transform articles into the expected format
      const transformedHeadlines = (data.articles || []).map((article: { title: string, url: string, sentiment_score: number }) => ({
        title: article.title,
        url: article.url,
        sentiment_score: article.sentiment_score
      }));

      // Update sentimentData with the backend response
      setSentimentData(prev => ({
        ...prev,
        [selectedStock]: {
          ...prev[selectedStock as keyof typeof prev],
          score: scaledScore,
          label: getSentimentLabel(scaledScore),
          sources: {
            news: scaledScore, // Use the overall sentiment score for news since backend only provides news
            social: 0, // Not provided by backend
            analyst: 0, // Not provided by backend
          },
          newsHeadlines: transformedHeadlines.length > 0 ? transformedHeadlines : prev[selectedStock as keyof typeof prev].newsHeadlines
        }
      }));
      
      toast({
        title: "Sentiment Analysis Complete",
        description: `Successfully analyzed sentiment for ${selectedStock}`,
      });
    } catch (error) {
      console.error('Error fetching sentiment data:', error);
      toast({
        title: "API Connection Error",
        description: "Could not connect to sentiment analysis backend. Using fallback data.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getSentimentLabel = (score: number) => {
    if (score >= 0.7) return 'Very Positive';
    if (score >= 0.5) return 'Positive';
    if (score >= 0.4) return 'Neutral';
    if (score >= 0.2) return 'Slightly Negative';
    return 'Negative';
  };

  const getSentimentColor = (score: number) => {
    if (score >= 0.7) return 'text-success';
    if (score >= 0.5) return 'text-primary';
    if (score >= 0.4) return 'text-accent';
    return 'text-destructive';
  };
  
  const getSentimentBarColor = (score: number) => {
    if (score >= 0.7) return 'bg-success';
    if (score >= 0.5) return 'bg-primary';
    if (score >= 0.4) return 'bg-accent';
    return 'bg-destructive';
  };

  const handleHeadlineClick = (url: string, e: React.MouseEvent<HTMLAnchorElement>) => {
    if (!url) {
      e.preventDefault();
      console.log("No URL provided for headline");
      return;
    }
    console.log(`Navigating to: ${url}`);
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      
      <main className="flex-grow">
        <section className="container px-4 py-12 md:py-16">
          <div className="max-w-4xl mx-auto">
            <div className="mb-8 animate-slide-down">
              <h1 className="text-3xl md:text-4xl font-bold mb-4 gradient-text">
                Market Sentiment Analysis
              </h1>
              <p className="text-lg text-muted-foreground">
                Real-time sentiment analysis from news, social media, and analyst reports
              </p>
            </div>
            
            <div className="mb-8 animate-scale-in">
              <Card className="glass-card p-6">
                <div className="mb-6 space-y-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Select Stock for Sentiment Analysis</label>
                    <Select value={selectedStock} onValueChange={setSelectedStock}>
                      <SelectTrigger className="w-full md:w-72 bg-secondary border-primary/30 hover:border-primary transition-all">
                        <SelectValue placeholder="Select a stock..." />
                      </SelectTrigger>
                      <SelectContent className="bg-secondary border-primary/30 text-foreground">
                        {stockOptions.map((option) => (
                          <SelectItem key={option.value} value={option.value} className="hover:bg-muted focus:bg-muted">
                            {option.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="flex gap-2">
                    <Button 
                      onClick={fetchSentimentData}
                      className="bg-primary hover:bg-primary/90 transition-all"
                      disabled={isLoading}
                    >
                      {isLoading ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <BarChart className="h-4 w-4 mr-2" />
                          Analyze Sentiment
                        </>
                      )}
                    </Button>
                    
                    <Button 
                      variant="outline" 
                      onClick={() => setShowConfig(!showConfig)}
                      className="border-primary/30"
                    >
                      {showConfig ? 'Hide Config' : 'API Config'}
                    </Button>
                  </div>
                  
                  {showConfig && (
                    <div className="p-3 border border-border rounded-md bg-secondary/30">
                      <label className="text-sm font-medium block mb-1">Backend API URL</label>
                      <div className="flex gap-2">
                        <Input 
                          value={backendUrl} 
                          onChange={(e) => setBackendUrl(e.target.value)}
                          placeholder="http://localhost:8000"
                          className="bg-background/50"
                        />
                        <Button 
                          variant="outline" 
                          onClick={() => {
                            toast({
                              title: "API URL Updated",
                              description: `Set to ${backendUrl}`,
                            });
                          }}
                        >
                          Save
                        </Button>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        Enter the URL where your Python sentiment analysis API is running
                      </p>
                    </div>
                  )}
                </div>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="space-y-6">
                    <div className="bg-secondary/50 p-4 rounded-lg border border-border">
                      <div className="flex justify-between items-center mb-4">
                        <h3 className="font-semibold flex items-center gap-2">
                          <BarChart className="h-5 w-5 text-accent" />
                          Overall Sentiment
                        </h3>
                        <div className={`${getSentimentColor(sentimentData[selectedStock as keyof typeof sentimentData].score)} font-bold text-lg`}>
                          {sentimentData[selectedStock as keyof typeof sentimentData].label}
                        </div>
                      </div>
                      
                      <p className="text-sm text-muted-foreground mb-2">Sentiment Score: {(sentimentData[selectedStock as keyof typeof sentimentData].score * 2 - 1).toFixed(4)}</p>
                      <div className="h-4 bg-muted rounded-full overflow-hidden">
                        <div 
                          className={`h-full ${getSentimentBarColor(sentimentData[selectedStock as keyof typeof sentimentData].score)} transition-all duration-1000 ease-out`}
                          style={{ 
                            width: animateScore ? 
                              `${sentimentData[selectedStock as keyof typeof sentimentData].score * 100}%` : '0%' 
                          }}
                        ></div>
                      </div>
                      
                      <div className="flex justify-between mt-1 text-xs text-muted-foreground">
                        <span>-1.0</span>
                        <span>0.0</span>
                        <span>1.0</span>
                      </div>
                    </div>
                    
                    <div className="bg-secondary/50 p-4 rounded-lg border border-border">
                      <h3 className="font-semibold mb-3">Sentiment by Source</h3>
                      <div className="space-y-3">
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span>News Articles</span>
                            <span className={getSentimentColor(sentimentData[selectedStock as keyof typeof sentimentData].sources.news)}>
                              {Math.round(sentimentData[selectedStock as keyof typeof sentimentData].sources.news * 100)}%
                            </span>
                          </div>
                          <div className="h-2 bg-muted rounded-full overflow-hidden">
                            <div 
                              className={`h-full ${getSentimentBarColor(sentimentData[selectedStock as keyof typeof sentimentData].sources.news)} transition-all duration-1000 ease-out`}
                              style={{ 
                                width: animateSources ? 
                                  `${sentimentData[selectedStock as keyof typeof sentimentData].sources.news * 100}%` : '0%' 
                              }}
                            ></div>
                          </div>
                        </div>
                        
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span>Social Media</span>
                            <span className={getSentimentColor(sentimentData[selectedStock as keyof typeof sentimentData].sources.social)}>
                              {sentimentData[selectedStock as keyof typeof sentimentData].sources.social === 0 ? 'N/A' : `${Math.round(sentimentData[selectedStock as keyof typeof sentimentData].sources.social * 100)}%`}
                            </span>
                          </div>
                          <div className="h-2 bg-muted rounded-full overflow-hidden">
                            <div 
                              className={`h-full ${getSentimentBarColor(sentimentData[selectedStock as keyof typeof sentimentData].sources.social)} transition-all duration-1000 ease-out`}
                              style={{ 
                                width: animateSources ? 
                                  `${sentimentData[selectedStock as keyof typeof sentimentData].sources.social * 100}%` : '0%' 
                              }}
                            ></div>
                          </div>
                        </div>
                        
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span>Analyst Reports</span>
                            <span className={getSentimentColor(sentimentData[selectedStock as keyof typeof sentimentData].sources.analyst)}>
                              {sentimentData[selectedStock as keyof typeof sentimentData].sources.analyst === 0 ? 'N/A' : `${Math.round(sentimentData[selectedStock as keyof typeof sentimentData].sources.analyst * 100)}%`}
                            </span>
                          </div>
                          <div className="h-2 bg-muted rounded-full overflow-hidden">
                            <div 
                              className={`h-full ${getSentimentBarColor(sentimentData[selectedStock as keyof typeof sentimentData].sources.analyst)} transition-all duration-1000 ease-out`}
                              style={{ 
                                width: animateSources ? 
                                  `${sentimentData[selectedStock as keyof typeof sentimentData].sources.analyst * 100}%` : '0%' 
                              }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-6">
                    <div className="bg-secondary/50 p-4 rounded-lg border border-border">
                      <h3 className="font-semibold mb-3 flex items-center gap-2">
                        <FileText className="h-5 w-5 text-primary" />
                        Key Topics & Keywords
                      </h3>
                      <div className="flex flex-wrap gap-2 mt-2">
                        {sentimentData[selectedStock as keyof typeof sentimentData].keywords.map((keyword, index) => (
                          <span 
                            key={index} 
                            className="px-2 py-1 text-xs rounded-full bg-muted text-foreground hover:bg-primary/20 transition-colors"
                          >
                            {keyword}
                          </span>
                        ))}
                      </div>
                    </div>
                    
                    <div className="bg-secondary/50 p-4 rounded-lg border border-border">
                      <h3 className="font-semibold mb-3">Latest News Headlines</h3>
                      <div className="space-y-2">
                        {sentimentData[selectedStock as keyof typeof sentimentData].newsHeadlines.map((headline, index) => (
                          <div 
                            key={index}
                            className="p-2 rounded hover:bg-muted transition-colors flex justify-between items-center"
                          >
                            <a 
                              href={headline.url} 
                              target="_blank" 
                              rel="noopener noreferrer" 
                              className="text-sm text-foreground hover:underline"
                              onClick={(e) => handleHeadlineClick(headline.url, e)}
                            >
                              {headline.title}
                            </a>
                            <span 
                              className={`text-xs px-2 py-0.5 rounded-full ${
                                headline.sentiment_score > 0 ? 'bg-success/20 text-success' : 
                                headline.sentiment_score < 0 ? 'bg-destructive/20 text-destructive' : 
                                'bg-accent/20 text-accent'
                              }`}
                            >
                              {headline.sentiment_score.toFixed(3)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    <div className="bg-secondary/50 p-4 rounded-lg border border-border">
                      <h3 className="font-semibold mb-3 flex items-center gap-2">
                        <AlertCircle className="h-5 w-5 text-accent" />
                        API Status
                      </h3>
                      <div className="flex items-center gap-2">
                        <div className={`h-3 w-3 rounded-full ${isLoading ? 'bg-accent animate-pulse' : backendUrl ? 'bg-primary' : 'bg-destructive'}`}></div>
                        <p className="text-sm">
                          {isLoading ? 'Fetching data...' : (backendUrl ? 'Ready to connect' : 'No API configured')}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            </div>
            
            <div className="flex justify-center mt-8 animate-fade-in">
              <Button asChild className="bg-gradient-to-r from-primary to-accent hover:brightness-110 transition-all">
                <Link to="/predict" state={{ stock: selectedStock, sentimentScore: sentimentData[selectedStock as keyof typeof sentimentData].score * 2 - 1 }} className="flex items-center gap-2">
                  <ChartLine className="h-5 w-5" />
                  <span>Proceed to Price Prediction</span>
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default SentimentPage;