export const cache_keys = {
  USER_STOCKS: "user:stocks",
  USER_FULL_NAME: "user:full_name",
  USER_STOCK_TRANSACTION: "user:stock:transaction:",
  STOCK_DATA: "stock:data:",
  STOCK_DATA_REALTIME: "stock:data:realtime:",
  STOCK_PREDICTION: "stock:predictions:",
};

export enum actions {
  SET_USER,
  SET_USER_FULL_NAME,
  SET_STOCK_HISTORY,
  SET_STOCK_PRICE,
  SET_USER_STOCK_TRANSACTIONS,
  SET_PREDICTION,
  SET_PREDICTION_VIEW_TIME,
  SET_PREDICTION_VIEW_MODEL,
}
