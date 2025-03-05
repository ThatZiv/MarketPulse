export type PredictionDatapoint = Record<string, string | number>;

export interface GlobalState {
  user: {
    id: string;
    email: string;
    name: string;
  };
  stocks: { [stock_ticker: string]: [] };
  predictions: { [stock_ticker: string]: PredictionDatapoint[] };
  views: {
    predictions: {
      timeWindow: number;
      model: string | null;
    };
  };
}
