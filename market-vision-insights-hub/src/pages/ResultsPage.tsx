
import React, { useEffect, useState } from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Link, useNavigate } from 'react-router-dom';
import { ArrowLeft, TrendingUp, Info, ChartLine, CheckCircle } from "lucide-react";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
  Tooltip
} from "recharts";

interface PredictionData {
  dates: string[];
  actualPrices: number[];
  predictions: number[];
  futureDates: string[];
  futurePredictions: number[];
  accuracy: number;
  sentiment: number;
}

// Demo data for visualization purposes
const demoData: Record<string, PredictionData> = {
  'MSFT': {
    dates: ['2025-04-23', '2025-04-24', '2025-04-25', '2025-04-26', '2025-04-27'],
    actualPrices: [410.22, 408.67, 412.91, 415.88, 414.30],
    predictions: [409.72, 407.97, 413.51, 416.28, 413.80],
    futureDates: ['2025-04-29', '2025-04-30', '2025-05-01', '2025-05-02', '2025-05-05'],
    futurePredictions: [416.22, 418.67, 422.91, 419.88, 425.30],
    accuracy: 94.7,
    sentiment: 0.72
  },
  'AAPL': {
    dates: ['2025-04-23', '2025-04-24', '2025-04-25', '2025-04-26', '2025-04-27'],
    actualPrices: [178.65, 180.22, 181.10, 180.79, 182.45],
    predictions: [179.25, 180.82, 180.50, 181.19, 183.05],
    futureDates: ['2025-04-29', '2025-04-30', '2025-05-01', '2025-05-02', '2025-05-05'],
    futurePredictions: [183.65, 185.22, 187.10, 186.79, 189.45],
    accuracy: 93.2,
    sentiment: 0.84
  },
  'TSLA': {
    dates: ['2025-04-23', '2025-04-24', '2025-04-25', '2025-04-26', '2025-04-27'],
    actualPrices: [171.22, 169.45, 172.90, 175.32, 177.65],
    predictions: [170.52, 168.95, 173.50, 174.82, 178.25],
    futureDates: ['2025-04-29', '2025-04-30', '2025-05-01', '2025-05-02', '2025-05-05'],
    futurePredictions: [179.22, 172.45, 168.90, 171.32, 174.65],
    accuracy: 90.5,
    sentiment: 0.48
  },
  'NFLX': {
    dates: ['2025-04-23', '2025-04-24', '2025-04-25', '2025-04-26', '2025-04-27'],
    actualPrices: [620.18, 618.45, 624.90, 629.75, 627.30],
    predictions: [621.28, 617.55, 625.30, 628.85, 626.50],
    futureDates: ['2025-04-29', '2025-04-30', '2025-05-01', '2025-05-02', '2025-05-05'],
    futurePredictions: [624.18, 631.45, 637.90, 629.75, 642.30],
    accuracy: 91.8,
    sentiment: 0.35
  },
  'GOOGL': {
    dates: ['2025-04-23', '2025-04-24', '2025-04-25', '2025-04-26', '2025-04-27'],
    actualPrices: [170.25, 172.50, 171.75, 173.20, 174.85],
    predictions: [169.85, 173.10, 171.25, 173.80, 175.25],
    futureDates: ['2025-04-29', '2025-04-30', '2025-05-01', '2025-05-02', '2025-05-05'],
    futurePredictions: [173.25, 176.50, 178.75, 176.20, 179.85],
    accuracy: 92.6,
    sentiment: 0.62
  },
  'AMZN': {
    dates: ['2025-04-23', '2025-04-24', '2025-04-25', '2025-04-26', '2025-04-27'],
    actualPrices: [178.35, 180.60, 179.20, 181.45, 183.75],
    predictions: [178.95, 181.20, 178.60, 182.05, 184.35],
    futureDates: ['2025-04-29', '2025-04-30', '2025-05-01', '2025-05-02', '2025-05-05'],
    futurePredictions: [182.35, 185.60, 188.20, 186.45, 190.75],
    accuracy: 91.9,
    sentiment: 0.55
  },
};

const ResultsPage = () => {
  const [stock, setStock] = useState<string>("");
  const [predictionData, setPredictionData] = useState<PredictionData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [showHistorical, setShowHistorical] = useState<boolean>(false);
  const [animateChart, setAnimateChart] = useState<boolean>(false);
  const navigate = useNavigate();

  useEffect(() => {
    // Get selected stock from session storage
    const selectedStock = sessionStorage.getItem('predictedStock');
    
    if (!selectedStock) {
      // If no stock was selected, redirect back to predict page
      navigate('/predict');
      return;
    }
    
    setStock(selectedStock);
    
    // Simulate loading prediction data
    setTimeout(() => {
      setPredictionData(demoData[selectedStock]);
      setIsLoading(false);
      
      // Animate the chart after data is loaded
      setTimeout(() => {
        setAnimateChart(true);
      }, 500);
    }, 1000);
  }, [navigate]);

  const getPriceChangeClass = (current: number, previous: number) => {
    const change = current - previous;
    return change >= 0 ? 'text-success' : 'text-destructive';
  };

  const getAccuracyColorClass = (accuracy: number) => {
    if (accuracy >= 95) return 'text-success';
    if (accuracy >= 90) return 'text-primary';
    if (accuracy >= 80) return 'text-accent';
    return 'text-destructive';
  };
  
  const getPriceChangeIcon = (current: number, previous: number) => {
    const change = current - previous;
    if (change >= 0) return '▲';
    return '▼';
  };

  // Create chart data
  const getChartData = (data: PredictionData) => {
    // Historical data
    const historical = data.dates.map((date, i) => ({
      date,
      actual: data.actualPrices[i],
      predicted: data.predictions[i]
    }));
    
    // Future predictions
    const future = data.futureDates.map((date, i) => ({
      date,
      predicted: data.futurePredictions[i],
      // No actual value for future dates
      actual: null
    }));
    
    return [...historical, ...future];
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Header />
        <main className="flex-grow flex items-center justify-center">
          <div className="text-center">
            <div className="h-16 w-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <h2 className="text-xl font-medium gradient-text">Processing Prediction Results</h2>
            <p className="text-muted-foreground mt-2">Integrating sentiment analysis with ARIMA-LSTM models...</p>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  if (!predictionData) {
    return (
      <div className="min-h-screen flex flex-col bg-background">
        <Header />
        <main className="flex-grow">
          <div className="container px-4 py-12 text-center">
            <h2 className="text-2xl font-bold mb-4 text-destructive">Error Loading Results</h2>
            <p className="mb-6 text-muted-foreground">We couldn't load the prediction data. Please try again.</p>
            <Button asChild className="bg-primary hover:bg-primary/90">
              <Link to="/predict">Return to Prediction Page</Link>
            </Button>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  const chartData = getChartData(predictionData);

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      
      <main className="flex-grow">
        <section className="container px-4 py-12 md:py-16">
          <div className="max-w-4xl mx-auto">
            <div className="mb-8 animate-slide-down">
              <Button asChild variant="outline" className="mb-4 border-primary/30 hover:border-primary/70">
                <Link to="/predict" className="flex items-center gap-2">
                  <ArrowLeft className="h-4 w-4" />
                  Back to Prediction
                </Link>
              </Button>
              
              <div className="flex flex-col md:flex-row md:items-center justify-between">
                <h1 className="text-3xl md:text-4xl font-bold mb-2 gradient-text">
                  {stock} Price Forecast
                </h1>
                
                <div className="flex items-center gap-2 mt-2 md:mt-0">
                  <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                    predictionData.sentiment >= 0.7 ? 'bg-success/20 text-success' :
                    predictionData.sentiment >= 0.5 ? 'bg-primary/20 text-primary' :
                    predictionData.sentiment >= 0.4 ? 'bg-accent/20 text-accent' : 
                    'bg-destructive/20 text-destructive'
                  }`}>
                    Sentiment: {Math.round(predictionData.sentiment * 100)}%
                  </div>
                  
                  <div className="px-3 py-1 rounded-full bg-primary/20 text-primary text-sm font-medium">
                    R²: {predictionData.accuracy}%
                  </div>
                </div>
              </div>
            </div>
            
            <Card className="glass-card p-6 mb-8 animate-scale-in">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <ChartLine className="h-5 w-5 text-primary" />
                Stock Price Prediction Chart
              </h2>
              
              <div className={`h-[350px] transition-opacity duration-1000 ${animateChart ? 'opacity-100' : 'opacity-0'}`}>
                <ChartContainer config={{
                  actual: { label: "Actual Price", color: "#10b981" },
                  predicted: { label: "Predicted Price", color: "#0ea5e9" }
                }} className="chart-container">
                  <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis 
                      dataKey="date" 
                      stroke="#cbd5e1"
                      tick={{ fill: '#cbd5e1' }}
                      tickFormatter={(value) => value.split('-').slice(1).join('/')}
                    />
                    <YAxis 
                      stroke="#cbd5e1"
                      tick={{ fill: '#cbd5e1' }}
                      domain={['dataMin - 10', 'dataMax + 10']}
                    />
                    <Tooltip content={<ChartTooltipContent />} />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="actual" 
                      stroke="#10b981" 
                      strokeWidth={2}
                      dot={{ r: 4, strokeWidth: 1 }}
                      activeDot={{ r: 6 }}
                      name="Actual Price"
                      isAnimationActive={animateChart}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="predicted" 
                      stroke="#0ea5e9" 
                      strokeWidth={2}
                      strokeDasharray="3 3"
                      dot={{ r: 4, strokeWidth: 1 }}
                      activeDot={{ r: 6 }}
                      name="Predicted Price"
                      isAnimationActive={animateChart}
                    />
                  </LineChart>
                </ChartContainer>
              </div>
              
              <div className="flex justify-between mt-4 text-xs text-muted-foreground">
                <div>
                  <span className="inline-block w-3 h-3 bg-success rounded-full mr-1"></span>
                  Actual Historical Prices
                </div>
                <div>
                  <span className="inline-block w-3 h-3 bg-primary rounded-full mr-1"></span>
                  Model Predictions (incl. future)
                </div>
              </div>
              
              <div className="mt-4 flex justify-center">
                <Button 
                  variant="outline" 
                  size="sm"
                  className="border-primary/30 hover:border-primary/70 text-xs"
                  onClick={() => setShowHistorical(!showHistorical)}
                >
                  {showHistorical ? 'Show Future Predictions' : 'Show Historical Accuracy'}
                </Button>
              </div>
            </Card>
            
            <div className="grid gap-8 md:grid-cols-2 animate-slide-up">
              {showHistorical ? (
                <Card className="glass-card p-6">
                  <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                    <CheckCircle className="h-5 w-5 text-success" />
                    Historical Accuracy
                  </h2>
                  
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                      <thead>
                        <tr className="border-b border-border">
                          <th className="py-3 px-4 text-left text-sm font-medium text-muted-foreground">Date</th>
                          <th className="py-3 px-4 text-right text-sm font-medium text-muted-foreground">Actual</th>
                          <th className="py-3 px-4 text-right text-sm font-medium text-muted-foreground">Predicted</th>
                          <th className="py-3 px-4 text-right text-sm font-medium text-muted-foreground">Error %</th>
                        </tr>
                      </thead>
                      <tbody>
                        {predictionData.dates.map((date, index) => {
                          const actual = predictionData.actualPrices[index];
                          const predicted = predictionData.predictions[index];
                          const error = Math.abs((predicted - actual) / actual) * 100;
                          
                          return (
                            <tr key={date} className="border-b border-border hover:bg-secondary/30">
                              <td className="py-3 px-4 text-sm">{date}</td>
                              <td className="py-3 px-4 text-right font-medium">${actual.toFixed(2)}</td>
                              <td className="py-3 px-4 text-right font-medium">${predicted.toFixed(2)}</td>
                              <td className={`py-3 px-4 text-right font-medium ${error < 1 ? 'text-success' : error < 2 ? 'text-primary' : 'text-accent'}`}>
                                {error.toFixed(2)}%
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </Card>
              ) : (
                <Card className="glass-card p-6">
                  <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                    <TrendingUp className="h-5 w-5 text-primary" />
                    5-Day Price Forecast
                  </h2>
                  
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                      <thead>
                        <tr className="border-b border-border">
                          <th className="py-3 px-4 text-left text-sm font-medium text-muted-foreground">Date</th>
                          <th className="py-3 px-4 text-right text-sm font-medium text-muted-foreground">Price ($)</th>
                          <th className="py-3 px-4 text-right text-sm font-medium text-muted-foreground">Change</th>
                        </tr>
                      </thead>
                      <tbody>
                        {predictionData.futureDates.map((date, index) => {
                          const price = predictionData.futurePredictions[index];
                          const prevPrice = index > 0 ? predictionData.futurePredictions[index - 1] : 
                            predictionData.actualPrices[predictionData.actualPrices.length - 1];
                          const change = ((price - prevPrice) / prevPrice) * 100;
                          
                          return (
                            <tr key={date} className="border-b border-border hover:bg-secondary/30">
                              <td className="py-3 px-4 text-sm">{date}</td>
                              <td className="py-3 px-4 text-right font-medium">${price.toFixed(2)}</td>
                              <td className={`py-3 px-4 text-right font-medium ${getPriceChangeClass(price, prevPrice)}`}>
                                {getPriceChangeIcon(price, prevPrice)} {Math.abs(change).toFixed(2)}%
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </Card>
              )}
              
              <div className="space-y-6">
                <Card className="glass-card p-6">
                  <h2 className="text-xl font-semibold mb-4">Model Performance</h2>
                  
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-muted-foreground">R² Score (Accuracy)</span>
                    <span className={`text-lg font-bold ${getAccuracyColorClass(predictionData.accuracy)}`}>
                      {predictionData.accuracy}%
                    </span>
                  </div>
                  
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div 
                      className={`h-full ${
                        predictionData.accuracy >= 95 ? 'bg-success' :
                        predictionData.accuracy >= 90 ? 'bg-primary' :
                        'bg-accent'
                      } transition-all duration-1500 ease-out`}
                      style={{ width: animateChart ? `${predictionData.accuracy}%` : '0%' }}
                    ></div>
                  </div>
                  
                  <div className="mt-4 text-sm text-muted-foreground">
                    <div className="flex items-start gap-2">
                      <Info className="h-4 w-4 mt-0.5 min-w-4 text-muted-foreground" />
                      <p>
                        The R² score indicates how well the model's predictions match actual historical data.
                        Our hybrid model integrates sentiment analysis with traditional prediction methods for improved accuracy.
                      </p>
                    </div>
                  </div>
                </Card>
                
                <Card className="glass-card p-6">
                  <h2 className="text-xl font-semibold mb-4">Prediction Confidence</h2>
                  
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-muted-foreground">Market Volatility</span>
                        <div className="h-2 w-2 rounded-full bg-primary"></div>
                      </div>
                      <div className="text-2xl font-bold mt-1">Medium</div>
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-muted-foreground">Price Momentum</span>
                        {predictionData.futurePredictions[4] > predictionData.futurePredictions[0] ? (
                          <div className="h-2 w-2 rounded-full bg-success"></div>
                        ) : (
                          <div className="h-2 w-2 rounded-full bg-destructive"></div>
                        )}
                      </div>
                      <div className="text-2xl font-bold mt-1">
                        {predictionData.futurePredictions[4] > predictionData.futurePredictions[0] ? 'Positive' : 'Negative'}
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-6 text-sm text-muted-foreground">
                    <p>
                      This prediction incorporates sentiment analysis (score: {Math.round(predictionData.sentiment * 100)}%) from news and social media as a key input to the ARIMA-LSTM hybrid model.
                    </p>
                  </div>
                </Card>
              </div>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default ResultsPage;
