export interface GlobalState {
  user: {
    id: string;
    email: string;
    name: string;
  };
  stocks: { [stock_ticker: string]: StockState };
  predictions: { [stock_ticker: string]: StockPrediction };
}

interface StockState {
  stock_name: string;
  stock_current_price: number;
  history: Array<number>;
  timestamp: number;
}

interface StockPrediction {
  model_name: string;
  prediction: Array<number>;
  timestamp: number;
}
