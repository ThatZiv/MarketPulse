/* eslint-disable @typescript-eslint/no-explicit-any */
import dataHandler from "@/lib/dataHandler";
import { actions } from "@/lib/constants";
import { type SupabaseClient } from "@supabase/supabase-js";
import { type IApi } from "@/lib/api";
import { type GlobalDispatch } from "@/lib/GlobalProvider";
import { UserStock, UserStockPurchase } from "@/types/stocks";

const mockDispatch: GlobalDispatch = jest.fn();

const createMockSupabase = (data: any, error: any = null) => {
  const mock = {
    from: jest.fn().mockReturnThis(),
    select: jest.fn().mockReturnThis(),
    eq: jest.fn().mockReturnThis(),
    order: jest.fn().mockReturnThis(),
    then: jest.fn().mockImplementation((callback) => {
      if (error) {
        callback({ data: null, error });
      } else {
        callback({ data, error: null });
      }
    }),
  };

  return mock as unknown as SupabaseClient & {
    from: jest.Mock;
    select: jest.Mock;
    eq: jest.Mock;
    order: jest.Mock;
  };
};

const createMockApi = (data: any) => ({
  getStockRealtime: jest.fn(() => Promise.resolve(data)),
});

describe("dataHandler", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });
  describe("forSupabase", () => {
    describe("getUserStocks", () => {
      it("should fetch user stocks and resolve with data", async () => {
        const mockUserStocks: UserStock[] = [
          {
            Stocks: {
              stock_ticker: "TICKER",
              stock_id: 0,
              stock_name: "Ticker Full Name",
            },
            shares_owned: 100,
            desired_investiture: 1000,
          },
        ];
        const mockSupabase = createMockSupabase(mockUserStocks);
        const handler = dataHandler().forSupabase(mockSupabase);
        const getUserStocks = handler.getUserStocks("user123");

        const result = await getUserStocks();

        expect(result).toEqual(mockUserStocks);
      });

      it("should reject on Supabase error", async () => {
        const mockError = new Error("Database error");
        const mockSupabase = createMockSupabase(null, mockError);
        const handler = dataHandler().forSupabase(mockSupabase);
        const getUserStocks = handler.getUserStocks("user123");

        await expect(getUserStocks()).rejects.toThrow(mockError);
      });
    });

    describe("getUserStockPurchases", () => {
      it("should fetch user stock purchases and resolve with data", async () => {
        const mockPurchases = [{ id: "1", amount: 100 }];
        const mockSupabase = createMockSupabase(mockPurchases);
        const handler = dataHandler().forSupabase(mockSupabase);
        const getUserStockPurchases = handler.getUserStockPurchases("user123");

        const result = await getUserStockPurchases();

        expect(mockSupabase.from).toHaveBeenCalledWith("User_Stock_Purchases");
        expect(mockSupabase.select).toHaveBeenCalledWith(
          "Stocks (stock_ticker), *"
        );
        expect(mockSupabase.eq).toHaveBeenCalledWith("user_id", "user123");
        expect(mockSupabase.order).toHaveBeenCalledWith("date", {
          ascending: false,
        });
        expect(result).toEqual(mockPurchases);
      });
    });

    describe("getUserStockPurchasesForStock", () => {
      it("should fetch purchases for a stock and dispatch if ticker provided", async () => {
        const mockPurchases: UserStockPurchase[] = [
          {
            user_id: "user123",
            date: "2023-01-01",
            Stocks: { stock_ticker: "TICKER" },
            price_purchased: 100,
            amount_purchased: 10,
          },
        ];
        const mockSupabase = createMockSupabase(mockPurchases);
        const handler = dataHandler(mockDispatch).forSupabase(mockSupabase);
        const getUserStockPurchasesForStock =
          handler.getUserStockPurchasesForStock(
            "user123",
            "stock123",
            "TICKER"
          );

        const result = await getUserStockPurchasesForStock();

        expect(mockSupabase.eq).toHaveBeenCalledWith("stock_id", "stock123");
        expect(mockSupabase.order).toHaveBeenCalledWith("date", {
          ascending: true,
        });
        expect(mockDispatch).toHaveBeenCalledWith({
          type: actions.SET_USER_STOCK_TRANSACTIONS,
          payload: { stock_ticker: "TICKER", data: mockPurchases },
        });
        expect(result).toEqual(mockPurchases);
      });

      it("should not dispatch if stock_ticker is missing", async () => {
        const mockSupabase = createMockSupabase([]);
        const handler = dataHandler(mockDispatch).forSupabase(mockSupabase);
        const getUserStockPurchasesForStock =
          handler.getUserStockPurchasesForStock("user123", "stock123");

        await getUserStockPurchasesForStock();

        expect(mockDispatch).not.toHaveBeenCalled();
      });
    });

    describe("getAllStocks", () => {
      it("should fetch all stocks and resolve with data", async () => {
        const mockStocks = [
          {
            stock_id: 1,
            stock_ticker: "TICKER",
            stock_name: "Test Stock",
            search: "test",
          },
        ];
        const mockSupabase = createMockSupabase(mockStocks);
        const handler = dataHandler().forSupabase(mockSupabase);
        const getAllStocks = handler.getAllStocks();

        const result = await getAllStocks();

        expect(mockSupabase.from).toHaveBeenCalledWith("Stocks");
        expect(mockSupabase.select).toHaveBeenCalledWith("*");
        expect(result).toEqual(mockStocks);
      });
    });
  });

  describe("forApi", () => {
    beforeEach(() => {
      jest.clearAllMocks();
    });
    describe("getStockRealtime", () => {
      it("should fetch realtime data and dispatch latest price", async () => {
        const mockData = [
          { time_stamp: ["2023-01-01", "12:00:00"], stock_close: 100 },
          { time_stamp: ["2023-01-01", "12:05:00"], stock_close: 105 },
        ];
        const mockApi = createMockApi(mockData);
        const handler = dataHandler(mockDispatch).forApi(
          mockApi as unknown as IApi
        );
        const getStockRealtime = handler.getStockRealtime("TICKER");

        const result = await getStockRealtime();

        expect(mockApi.getStockRealtime).toHaveBeenCalledWith("TICKER");
        expect(mockDispatch).toHaveBeenCalledWith({
          type: actions.SET_STOCK_PRICE,
          payload: {
            stock_ticker: "TICKER",
            data: 105,
            timestamp: new Date("2023-01-01 12:05:00").getTime(),
          },
        });
        expect(result).toEqual(mockData);
      });

      it("should not dispatch if no dispatch", async () => {
        const mockApi = createMockApi([
          { time_stamp: ["2023-01-01", "12:00:00"], stock_close: 100 },
          { time_stamp: ["2023-01-01", "12:05:00"], stock_close: 105 },
        ]);
        const handler = dataHandler().forApi(mockApi as unknown as IApi);
        const getStockRealtime = handler.getStockRealtime("TICKER");

        await getStockRealtime();

        expect(mockDispatch).not.toHaveBeenCalled();
      });
    });
  });
});
