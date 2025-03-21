import { type StockDataItem } from "./stocks";

export type PredictionDatapoint = Record<string, string | number>;
export type PurchaseHistoryDatapoint = {
  date: string;
  amount_purchased: number;
  price_purchased: number;
};

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
  history: { [stock_ticker: string]: PurchaseHistoryDatapoint[] };
  views: {
    predictions: {
      timeWindow: number;
      model: string | null;
    };
  };
}
