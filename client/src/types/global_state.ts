import { type StockDataItem } from "./stocks";

export type PredictionDatapoint = Record<string, string | number>;

export interface StocksState {
  stock_name: string;
  history: StockDataItem[];
  current_price?: number;
  timestamp: number;
}
export interface GlobalState {
  user: {
    id: string;
    email: string;
    name: string;
    url: string;
  };
  stocks: { [stock_ticker: string]: StocksState };
  predictions: { [stock_ticker: string]: PredictionDatapoint[] };
  views: {
    predictions: {
      timeWindow: number;
      model: string | null;
    };
  };
}
