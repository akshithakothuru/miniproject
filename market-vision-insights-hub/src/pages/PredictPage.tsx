import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import Header from '../components/Header';
import Footer from '../components/Footer';
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Loader2, TrendingUp, AlertCircle, BarChart, ChevronDown, ChevronUp, ArrowRight } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const PredictPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { toast } = useToast();
  
  // Extract stock, sentiment score, and sentiment data from location state
  const { stock = 'AAPL', sentimentScore = 0, sentimentData } = location.state || {};
  
  const [isLoading, setIsLoading] = useState(false);
  const [predictionData, setPredictionData] = useState<{
    ticker: string;
    predictions: { date: string; predicted_price: number }[];
    mae: number;
    mape: number;
    plot_url: string | null;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isHeadlinesExpanded, setIsHeadlinesExpanded] = useState(false);
  const backendUrl = 'http://localhost:8001'; // Price prediction backend

  const fetchPricePrediction = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${backendUrl}/predict-price?ticker=${stock}&sentiment_score=${sentimentScore}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch prediction: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('Price Prediction Response:', data);
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      setPredictionData(data);
      
      toast({
        title: "Price Prediction Complete",
        description: `Predicted prices for ${stock}: ${data.predictions.map((pred: any) => `${pred.date}: $${pred.predicted_price.toFixed(2)}`).join(', ')}`,
      });
    } catch (error: any) {
      console.error('Error fetching price prediction:', error);
      setError(error.message || "Could not fetch price prediction. Please try again.");
      toast({
        title: "Prediction Error",
        description: error.message || "Could not fetch price prediction. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleHeadlineClick = (url: string) => {
    console.log(`Navigating to: ${url}`);
    window.open(url, "_blank");
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      
      <main className="flex-grow">
        <section className="container px-4 py-12 md:py-16">
          <div className="max-w-4xl mx-auto">
            <div className="mb-8 animate-slide-down">
              <h1 className="text-3xl md:text-4xl font-bold mb-4 gradient-text">
                Price Prediction for {stock}
              </h1>
              <p className="text-lg text-muted-foreground">
                Review sentiment analysis and predict the future price of {stock}.
              </p>
            </div>

            {/* Sentiment Analysis Block */}
            {sentimentData && (
              <div className="mb-12 animate-scale-in">
                <Card className="glass-card p-6">
                  <div className="mb-6 space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="font-semibold mb-2">Overall Sentiment</h3>
                        <p className={`text-lg font-bold ${sentimentData.label === 'Positive' ? 'text-green-500' : sentimentData.label === 'Negative' ? 'text-red-500' : 'text-yellow-500'}`}>
                          {sentimentData.label}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Sentiment Score: {sentimentScore.toFixed(4)}
                        </p>
                      </div>
                      <div className="w-32 h-2 bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 rounded-full relative">
                        <div
                          className="absolute h-4 w-4 rounded-full bg-foreground border-2 border-background"
                          style={{ left: `calc(${(sentimentScore + 1) * 50}% - 8px)` }}
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <h4 className="font-semibold mb-2">Sentiment by Source</h4>
                        <ul className="text-sm space-y-1">
                          <li>
                            News Articles: <span className="font-medium">{sentimentData.sources?.news || 'N/A'}</span>
                          </li>
                          <li>
                            Social Media: <span className="font-medium">{sentimentData.sources?.social || 'N/A'}</span>
                          </li>
                          <li>
                            Analyst Reports: <span className="font-medium">{sentimentData.sources?.analyst || 'N/A'}</span>
                          </li>
                        </ul>
                      </div>
                      <div>
                        <h4 className="font-semibold mb-2">Key Topics & Keywords</h4>
                        <div className="flex flex-wrap gap-2">
                          {sentimentData.keywords?.map((keyword: string, index: number) => (
                            <span
                              key={index}
                              className="px-2 py-1 bg-secondary text-foreground rounded-full text-xs font-medium"
                            >
                              {keyword}
                            </span>
                          )) || <span className="text-sm text-muted-foreground">N/A</span>}
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-semibold mb-3 flex items-center gap-2">
                        Latest News Headlines
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setIsHeadlinesExpanded(!isHeadlinesExpanded)}
                        >
                          {isHeadlinesExpanded ? (
                            <ChevronUp className="h-4 w-4" />
                          ) : (
                            <ChevronDown className="h-4 w-4" />
                          )}
                        </Button>
                      </h4>
                      {(isHeadlinesExpanded ? sentimentData.headlines : sentimentData.headlines?.slice(0, 5))?.map(
                        (headline: { title: string; url: string; sentiment: number }, index: number) => (
                          <div
                            key={index}
                            className="flex items-center justify-between py-2 border-b border-border cursor-pointer hover:bg-secondary/50 transition-colors"
                            onClick={() => handleHeadlineClick(headline.url)}
                          >
                            <p className="text-sm text-foreground">{headline.title}</p>
                            <span
                              className={`text-xs font-medium px-2 py-1 rounded-full ${
                                headline.sentiment > 0
                                  ? 'bg-green-500/20 text-green-500'
                                  : headline.sentiment < 0
                                  ? 'bg-red-500/20 text-red-500'
                                  : 'bg-yellow-500/20 text-yellow-500'
                              }`}
                            >
                              {headline.sentiment.toFixed(3)}
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                </Card>
              </div>
            )}

            {/* Price Prediction Block */}
            <div className="mb-8 animate-scale-in">
              <Card className="glass-card p-6">
                <div className="mb-6 space-y-4">
                  <div>
                    <h3 className="font-semibold mb-2">Selected Stock: {stock}</h3>
                    <p className="text-sm text-muted-foreground">
                      Sentiment Score: {sentimentScore.toFixed(4)}
                    </p>
                  </div>
                  
                  <Button 
                    onClick={fetchPricePrediction}
                    className="bg-primary hover:bg-primary/90 transition-all"
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Predicting...
                      </>
                    ) : (
                      <>
                        <TrendingUp className="h-4 w-4 mr-2" />
                        Predict Price
                      </>
                    )}
                  </Button>
                </div>
                
                {error && (
                  <div className="bg-destructive/20 p-4 rounded-lg border border-destructive mb-6">
                    <h3 className="font-semibold mb-2 flex items-center gap-2 text-destructive">
                      <AlertCircle className="h-5 w-5" />
                      Error
                    </h3>
                    <p className="text-sm text-destructive">{error}</p>
                  </div>
                )}

                {predictionData && (
                  <div className="space-y-6">
                    <div className="bg-secondary/50 p-4 rounded-lg border border-border">
                      <h3 className="font-semibold mb-2 flex items-center gap-2">
                        <TrendingUp className="h-5 w-5 text-primary" />
                        Predicted Prices
                      </h3>
                      {predictionData.predictions.map((pred, index) => (
                        <p key={index} className="text-lg font-bold text-foreground">
                          {pred.date}: ${pred.predicted_price.toFixed(2)}
                        </p>
                      ))}
                    </div>
                    
                    <div className="bg-secondary/50 p-4 rounded-lg border border-border">
                      <h3 className="font-semibold mb-3 flex items-center gap-2">
                        <BarChart className="h-5 w-5 text-accent" />
                        Model Evaluation Metrics
                      </h3>
                      <div className="space-y-2">
                        <p className="text-sm">
                          MSE: {predictionData.mae.toFixed(2)}
                        </p>
                        <p className="text-sm">
                          MAPE: {predictionData.mape.toFixed(2)}%
                        </p>
                      </div>
                    </div>
                    
                    {predictionData.plot_url && (
                      <div className="bg-secondary/50 p-4 rounded-lg border border-border">
                        <h3 className="font-semibold mb-3 flex items-center gap-2">
                          <TrendingUp className="h-5 w-5 text-primary" />
                          Price Trend
                        </h3>
                        <img 
                          src={`${backendUrl}${predictionData.plot_url}?t=${new Date().getTime()}`} 
                          alt={`${stock} price trend`} 
                          className="w-full h-auto rounded-lg"
                        />
                      </div>
                    )}
                  </div>
                )}
                
                <div className="bg-secondary/50 p-4 rounded-lg border border-border mt-6">
                  <h3 className="font-semibold mb-3 flex items-center gap-2">
                    <AlertCircle className="h-5 w-5 text-accent" />
                    API Status
                  </h3>
                  <div className="flex items-center gap-2">
                    <div className={`h-3 w-3 rounded-full ${isLoading ? 'bg-accent animate-pulse' : error ? 'bg-destructive' : predictionData ? 'bg-primary' : 'bg-muted'}`}></div>
                    <p className="text-sm">
                      {isLoading ? 'Fetching prediction...' : error ? 'Failed to fetch prediction' : predictionData ? 'Prediction complete' : 'Ready to predict'}
                    </p>
                  </div>
                </div>
              </Card>
            </div>
            
            <div className="flex justify-center mt-8 animate-fade-in">
              <Button 
                onClick={() => navigate('/sentiment')}
                className="bg-gradient-to-r from-primary to-accent hover:brightness-110 transition-all"
              >
                Back to Sentiment Analysis
              </Button>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default PredictPage;