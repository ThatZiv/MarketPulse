import { Stock, type UserStock, type UserStockPurchase } from "@/types/stocks";
import { type SupabaseClient } from "@supabase/supabase-js";
import { type GlobalDispatch } from "./GlobalProvider";
import { IApi } from "./api";
import { actions } from "./constants";

/**
 * This factory for cached data (api or supabase) requests.
 * @param {GlobalDispatch} dispatch - global dispatch for caching if needed
 */
export default function dataHandler(dispatch?: GlobalDispatch) {
  return {
    /**
     * This factory for supabase requests.
     * @param supabase instance of SupabaseClient
     * @example
     * const supabase = dataHandler(dispatch).forSupabase(supabase);
     */
    forSupabase: (supabase: SupabaseClient) => ({
      /**
       * get user stocks
       * @param user_id - user_id string from client
       * @example
       * const supabase = dataHandler(dispatch).forSupabase(supabase);
       * const getUserStocks = supabase.getUserStocks(user_id);
       * const userStocks = await getUserStocks();
       * getUserStocks() // can be passed into queryFn
       */
      getUserStocks: (user_id: string) => (): Promise<UserStock[]> =>
        new Promise((resolve, reject) => {
          supabase
            .from("User_Stocks")
            .select("Stocks (*), shares_owned, desired_investiture")
            .eq("user_id", user_id)
            .order("created_at", { ascending: false })
            .then(({ data, error }) => {
              if (error) reject(error);
              // @ts-expect-error Stocks will never expand to an array
              resolve(data || []);
            });
        }),
      /**
       * get user stock purchases
       * @param user_id - user_id string from client
       * @example
       * const supabaseDataHandler = dataHandler(dispatch).forSupabase(supabase);
       * const getUserStockPurchases = supabase.getUserStockPurchases(user_id);
       * const userStockPurchases = await getUserStockPurchases();
       * */
      getUserStockPurchases:
        (user_id: string) => (): Promise<UserStockPurchase[]> =>
          new Promise((resolve, reject) => {
            supabase
              .from("User_Stock_Purchases")
              .select("Stocks (stock_ticker), *")
              .eq("user_id", user_id)
              .order("date", { ascending: false })
              .then(({ data, error }) => {
                if (error) {
                  reject(error);
                }
                resolve(data || []);
              });
          }),
      /**
       * get user stock purchases for a specific stock
       * @param user_id {string}
       * @param stock_id {string}
       * @param stock_ticker {string} - optional, used for caching
       * @example
       * const supabaseDataHandler = dataHandler(dispatch).forSupabase(supabase);
       * const getUserStockPurchasesForStock = supabase.getUserStockPurchasesForStock(user_id, stock_id);
       * const userStockPurchasesForStock = await getUserStockPurchasesForStock();
       */
      getUserStockPurchasesForStock:
        (user_id: string, stock_id: string, stock_ticker?: string) =>
        (): Promise<UserStockPurchase[]> =>
          new Promise((resolve, reject) => {
            supabase
              .from("User_Stock_Purchases")
              .select("Stocks (stock_ticker), *")
              .eq("user_id", user_id)
              .eq("stock_id", stock_id)
              .order("date", { ascending: true })
              .then(({ data, error }) => {
                if (error) {
                  reject(error);
                }
                if (dispatch && stock_ticker)
                  dispatch({
                    type: actions.SET_USER_STOCK_TRANSACTIONS,
                    payload: {
                      stock_ticker,
                      data,
                    },
                  });
                resolve(data as UserStockPurchase[]);
              });
          }),
      /**
       * Gets all available stocks
       * @example
       * const supabaseDataHandler = dataHandler(dispatch).forSupabase(supabase);
       * const getAllStocks = supabase.getAllStocks();
       * const allStocks = await getAllStocks();
       */
      getAllStocks: () => (): Promise<Stock[]> =>
        new Promise((resolve, reject) => {
          supabase
            .from("Stocks")
            .select("*")
            .then(({ data, error }) => {
              if (error) reject(error);
              return resolve(data || []);
            });
        }),
    }),
    /**
     * Factory for api requests
     * @param api - instance of our api wrapper
     * @example
     * const dh = dataHandler(dispatch);
     * const apiDataHandler = dh.forApi(api); // api is an instance of our global api wrapper
     */
    forApi: (api: IApi) => ({
      /**
       * get and cache stock realtime data
       * @param stock_ticker - stock_ticker string from client
       * @example
       * const apiCl = dataHandler(dispatch).forApi(api).getStockRealtime(stock_ticker);
       * const stockRealtime = await getStockRealtime();
       *  */
      getStockRealtime: (stock_ticker: string) => async () => {
        const data = await api.getStockRealtime(stock_ticker);
        if (!data) return [];
        if (dispatch)
          dispatch({
            type: actions.SET_STOCK_PRICE,
            payload: {
              stock_ticker,
              data: data[data.length - 1].stock_close,
              timestamp: new Date(
                data[data.length - 1].time_stamp.join(" ") + " UTC"
              ).getTime(),
            },
          });
        return data;
      },
    }),
  };
}
