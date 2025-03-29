export interface Stock {
  stock_id: number;
  stock_ticker: string;
  stock_name: string;
}

export interface StockDataItem {
  stock_id: number;
  sentiment_data: number;
  news_data: number;
  stock_volume: number;
  stock_close: number;
  stock_high: number;
  stock_low: number;
  stock_open: number;
  time_stamp: string[]; // FIXME this is a symptom of using timestampz
}

type Forecast = Array<number>;
type ModelName = string;
interface ModelOutput {
  forecast: Forecast;
  name: ModelName;
}

export interface StockPrediction {
  stock_id: number;
  created_at: string;
  output: ModelOutput[];
}

export interface UserStock {
  Stocks: {
    stock_id: number;
    stock_name: string;
    stock_ticker: string;
  };
  shares_owned: number;
  desired_investiture: number;
}

export interface UserStockPurchase {
  user_id: string;
  date: string;
  Stocks: {
    stock_ticker: string;
  };
  price_purchased: number;
  amount_purchased: number;
}
