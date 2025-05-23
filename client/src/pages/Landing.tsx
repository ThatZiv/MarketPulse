import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useSupabase } from "@/database/SupabaseProvider";
import { useApi } from "@/lib/ApiProvider";
import { useQueries, useQuery } from "@tanstack/react-query";
import { extractColors } from "extract-colors";
import { Link, useNavigate } from "react-router";
import { actions, cache_keys } from "@/lib/constants";
import { type PurchaseHistoryDatapoint } from "@/types/global_state";
import { useGlobal } from "@/lib/GlobalProvider";
import { PurchaseHistoryCalculator } from "@/lib/Calculator";
import React, { useMemo, useState } from "react";
import Marquee from "react-fast-marquee";
import { Button } from "@/components/ui/button";
import { ArrowRight, Dot, Pencil } from "lucide-react";
import InfoTooltip from "@/components/InfoTooltip";
import moment from "moment";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { LiaSortSolid } from "react-icons/lia";
import dataHandler from "@/lib/dataHandler";
import { UserStock } from "@/types/stocks";
import { IApi } from "@/lib/api";

interface StockCardProps {
  stock: UserStock;
  activeCard: number;
  setActiveCard: React.Dispatch<React.SetStateAction<number>>;
}

const { toDollar } = PurchaseHistoryCalculator;

export default function Landing() {
  const { supabase, displayName, user } = useSupabase();
  const api = useApi();
  const { dispatch } = useGlobal();
  const [activeCard, setActiveCard] = useState(-1);
  const [sort, setSort] = useState("None");
  const {
    state: { history },
  } = useGlobal();
  const {
    data: stocks,
    error: stocksError,
    status: stocksStatus,
  } = useQuery<UserStock[]>({
    queryKey: [cache_keys.USER_STOCKS],
    queryFn: dataHandler(dispatch)
      .forSupabase(supabase)
      .getUserStocks(user?.id ?? ""),
    enabled: !!user,
  });

  // global coroutine to cache in state (entrypoint)
  useQueries({
    queries: [
      // user's stock transactions
      {
        queryKey: [cache_keys.USER_STOCK_TRANSACTION, "global"],
        // refetchInterval: () => (import.meta.env.PROD ? 1000 * 60 * 5 : 0),
        queryFn: async () => {
          const data = await dataHandler(dispatch)
            .forSupabase(supabase)
            .getUserStockPurchases(user?.id ?? "")();
          const allTransactions: {
            [ticker: string]: Array<
              PurchaseHistoryDatapoint & {
                Stocks: { stock_ticker: string };
              }
            >;
          } = {};
          data?.forEach((transaction) => {
            if (!allTransactions[transaction.Stocks.stock_ticker]) {
              allTransactions[transaction.Stocks.stock_ticker] = [];
            }
            allTransactions[transaction.Stocks.stock_ticker].push(transaction);
          });
          Object.entries(allTransactions).forEach(([ticker, transactions]) => {
            dispatch({
              type: actions.SET_USER_STOCK_TRANSACTIONS,
              payload: { data: transactions, stock_ticker: ticker },
            });
          });

          return null;
        },
      },
      // realtime stock prices for each stock the user has
      ...(stocks
        ?.map((stock) => [
          {
            queryKey: [
              cache_keys.STOCK_DATA_REALTIME,
              stock.Stocks.stock_ticker,
            ],
            refetchInterval: () => (import.meta.env.PROD ? 1000 * 60 * 5 : 0),
            // staleTime: 1000 * 60 * 5, // as of now, we'll keep realtime prices
            queryFn: dataHandler(dispatch)
              .forApi(api as IApi)
              .getStockRealtime(stock.Stocks.stock_ticker),
            enabled: !!stock.Stocks.stock_ticker && !!api,
          },
          // this is less efficient but good for granular caching
          // {
          //   queryKey: [
          //     cache_keys.USER_STOCK_TRANSACTION,
          //     stock.Stocks.stock_id.toString(),
          //   ],
          //   queryFn: dataHandler(dispatch)
          //     .forSupabase(supabase)
          //     .getUserStockPurchasesForStock(
          //       user?.id ?? "",
          //       stock.Stocks.stock_id.toString()
          //     ),
          //   enabled: !!user,
          // },
        ])
        .flat() ?? []),
    ],
  });
  let sortedStocks: UserStock[] = [];
  const validStocks = Array.isArray(stocks) ? stocks : [];

  if (validStocks.length > 0) {
    sortedStocks = [...validStocks];

    if (sort === "A-Z") {
      sortedStocks.sort((item1, item2) =>
        item1.Stocks.stock_name.localeCompare(item2.Stocks.stock_name)
      );
    } else if (sort === "Z-A") {
      sortedStocks.sort((item1, item2) =>
        item2.Stocks.stock_name.localeCompare(item1.Stocks.stock_name)
      );
    } else if (sort === "Shares:L-H") {
      sortedStocks.sort((item1, item2) => {
        const sum1 =
          (history[item1.Stocks.stock_ticker] ?? []).reduce(
            (sum, stock) => sum + stock.amount_purchased,
            0
          ) || 0;
        const sum2 =
          (history[item2.Stocks.stock_ticker] ?? []).reduce(
            (sum, stock) => sum + stock.amount_purchased,
            0
          ) || 0;
        return sum1 - sum2;
      });
    } else if (sort === "Shares:H-L") {
      sortedStocks.sort((item1, item2) => {
        const sum1 =
          (history[item1.Stocks.stock_ticker] ?? []).reduce(
            (sum, stock) => sum + stock.amount_purchased,
            0
          ) || 0;
        const sum2 =
          (history[item2.Stocks.stock_ticker] ?? []).reduce(
            (sum, stock) => sum + stock.amount_purchased,
            0
          ) || 0;
        return sum2 - sum1;
      });
    }
  }

  // get stock images for each stock
  const stockImages = useQueries({
    queries:
      sortedStocks?.map((stock) => ({
        queryKey: ["stock_img", stock.Stocks.stock_ticker],
        queryFn: () => api?.getStockLogo(stock.Stocks.stock_ticker),
        staleTime: Infinity,
      })) || [],
  }).map((query) => query.data);

  // cache stock image colors (this isn't necessary to cache as its client side)
  const stockColors = useQueries({
    queries:
      stockImages?.map((img) => ({
        queryKey: ["stock_img_color", img],
        queryFn: () => extractColors(img ?? ""),
        staleTime: Infinity,
      })) || [],
  })
    ?.map((query) => query.data)
    .map(
      // sort by most common color first
      (img) => img?.sort((a, b) => b.area - a.area).map((color) => color.hex)
    );

  // used to unset card on mouse out to parent dom element
  const handleMouseOut = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      setActiveCard(-1);
    }
  };

  const loading = stocksStatus === "pending";

  if (stocksError) {
    return (
      <div className="flex flex-col justify-center items-center h-screen">
        <h1 className="text-3xl">Error</h1>
        <p>
          Unfortunately, we encountered an error fetching your stocks. Please
          refresh the page or try again later.
        </p>
      </div>
    );
  }
  return (
    <div onMouseOver={handleMouseOut} className="min-h-screen w-full">
      <h1 className="text-4xl text-center flex-1 tracking-tight">
        Welcome <b>{displayName || "User"}</b>
      </h1>
      <Separator className="my-2" />

      <div className="flex flex-col items-center gap-4 flex-grow">
        <section className="w-full">
          <h2 className="text-2xl font-light mb-6 text-center">
            Your Investment Portfolio
          </h2>
          {stocks?.length === 0 && (
            <div className="text-center text-gray-500 mb-4">
              No investments found, click the "+" to add your first investment
            </div>
          )}
          <div className="flex md:flex-row flex-col justify-center items-center gap-6">
            {loading ? (
              <>
                <Skeleton className="w-40 h-[100px]" />
                <Skeleton className="w-20 h-[100px]" />
                <Skeleton className="w-32 h-[100px]" />
              </>
            ) : (
              <div className="flex flex-col gap-6">
                <div className="flex flex-col justify-end items-end gap-6">
                  <div className="flex flex-col justify-center items-start">
                    <div className="flex items-center">
                      <LiaSortSolid className="ml-2" />
                      <h3 className="">Sort:</h3>
                    </div>
                    <Select value={sort} onValueChange={setSort}>
                      <SelectTrigger className="md:w-40 rounded-lg w-24 dark:border-white">
                        <SelectValue placeholder="None Selected" />
                      </SelectTrigger>
                      <SelectContent className="rounded-xl">
                        <SelectItem value="None" className="rounded-lg">
                          None
                        </SelectItem>
                        <SelectItem value="A-Z" className="rounded-lg">
                          A-Z
                        </SelectItem>
                        <SelectItem value="Z-A" className="rounded-lg">
                          Z-A
                        </SelectItem>
                        <SelectItem value="Shares:L-H" className="rounded-lg">
                          Current Shares Owned: Low to High
                        </SelectItem>
                        <SelectItem value="Shares:H-L" className="rounded-lg">
                          Current Shares Owned: High to Low
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div
                  onClick={handleMouseOut}
                  onMouseLeave={() => setActiveCard(-1)}
                  className="flex flex-row flex-wrap items-center justify-center gap-6"
                >
                  {sortedStocks?.map((stock, index) => (
                    <StockCard
                      key={stock?.Stocks?.stock_name}
                      activeCard={activeCard}
                      setActiveCard={setActiveCard}
                      stock={stock}
                      img={stockImages[index] ?? ""}
                      colors={stockColors[index] ?? []}
                    />
                  ))}
                </div>
              </div>
            )}
            <Link
              className="flex items-center justify-center w-20 h-20 text-white rounded-full pb-1 bg-primary text-4xl font-bold shadow hover:shadow-md transition-transform transform hover:scale-105 active:scale-95"
              to="/stocks"
            >
              +
            </Link>
          </div>
        </section>
      </div>
    </div>
  );
}

// component for stock cards on landing page
function StockCard({
  stock,
  img,
  colors,
  activeCard,
  setActiveCard,
}: StockCardProps & { img: string; colors: string[] }) {
  const {
    state: { history, stocks },
  } = useGlobal();

  const navigate = useNavigate();
  const ticker = stock.Stocks.stock_ticker;
  const userStockHistory = history[ticker];
  const thisStock = stocks[ticker];
  const calc = useMemo(
    () => new PurchaseHistoryCalculator(userStockHistory ?? []),
    [userStockHistory]
  );

  const isShown = activeCard === stock.Stocks.stock_id;
  return (
    <span
      className={`${isShown && "w-[350px] "} `}
      onMouseEnter={() => setActiveCard(stock.Stocks.stock_id)}
    >
      <div
        className={`bg-white cursor-pointer hover:bg-slate-200 ${
          !isShown && "h-[200px]"
        } dark:hover:bg-gray-800 dark:bg-black p-6 rounded-xl shadow-lg hover:shadow-2xl transition-all transform duration-500 hover:scale-105 ease-in-out`}
        style={{
          border: `4px solid ${colors[0]}`,
        }}
      >
        <div className="flex justify-center mb-4">
          <img
            src={img}
            alt={stock.Stocks.stock_name}
            className="w-20 h-20 object-cover rounded-lg shadow-md"
          />
        </div>

        <h3 className="text-xl font-semibold text-gray-900 dark:text-white uppercase tracking-wide mb-3">
          {ticker}
        </h3>
        {/* {isShown && (
          <div className="flex items-stretch flex-row justify-between ">
            <Button variant="outline" className="hover:invert" onClick={() => setActiveCard()}>
              <ArrowLeft />
            </Button>
            <Button variant="secondary" className="hover:invert">
              Next
            </Button>
          </div>
        )} */}
        {!isShown && (
          // marquee preview data
          <div className="max-w-[125px] text-xs">
            <Marquee
              pauseOnHover
              speed={15}
              className="text-center flex items-center justify-center gap-2"
            >
              <Dot className="text-gray-500" />
              {thisStock?.current_price && (
                <>
                  <span className="text-sm ">
                    {toDollar(thisStock.current_price)} per share,{" "}
                  </span>
                  {userStockHistory && (
                    <span className="text-sm ">
                      {toDollar(calc.getTotalValue(thisStock.current_price))}{" "}
                      total value,{" "}
                    </span>
                  )}
                </>
              )}

              {userStockHistory && (
                <>
                  <span className={`text-xs mx-1`}>
                    <span
                      className={`${
                        calc.getProfit() < 0 ? "text-red-600" : "text-green-600"
                      } text-sm`}
                    >
                      {calc.getProfit() < 0
                        ? toDollar(-1 * calc.getProfit())
                        : toDollar(calc.getProfit())}
                    </span>{" "}
                    {calc.getProfit() < 0 ? "loss" : "profit"},
                  </span>
                  <span className={`text-xs `}>
                    <span className="text-sm">{calc.getTotalShares()} </span>
                    shares,
                  </span>{" "}
                  {calc.getTotalShares() > 0 && (
                    <span className={`text-[8px] `}>
                      <span className="text-sm">
                        {toDollar(calc.getAveragePrice())}{" "}
                      </span>
                      average cost per share{" "}
                    </span>
                  )}
                </>
              )}
            </Marquee>
          </div>
        )}
        <div
          className={`flex flex-col items-center overflow-hidden transition-all duration-500 ease-in-out ${
            isShown ? "max-h-[1000px] opacity-100" : "max-h-0 opacity-0"
          }`}
        >
          {isShown && (
            <>
              {userStockHistory ? (
                <>
                  <Separator className="mb-4 border-2 dark:border-gray-300 border-gray-800" />
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {stock.Stocks.stock_name}
                  </p>
                  <div className="flex flex-col items-center w-full">
                    <div className="text-xs font-medium text-gray-600 dark:text-gray-300 inline">
                      <div className="flex items-center gap-1">
                        <span className="text-xl font-semibold text-gray-900 dark:text-white">
                          {calc.getTotalShares()}
                        </span>{" "}
                        share{calc.getTotalShares() === 1 ? "" : "s"} currently
                        owned
                        <InfoTooltip side="right">
                          This is the total number of shares you own for this
                          stock. It is calculated by taking the difference
                          between all your purchased stocks and those that were
                          sold.
                        </InfoTooltip>
                      </div>
                    </div>
                    <div className="text-xs font-medium text-gray-600 dark:text-gray-300 inline">
                      <div className="flex items-center gap-1">
                        {calc.getProfit() < 0 ? (
                          <div>
                            <span
                              className={`text-xl font-semibold $ text-red-600`}
                            >
                              {toDollar(calc.getProfit() * -1)}
                            </span>
                            <span className="inline"> last sale's losses</span>
                          </div>
                        ) : (
                          <div>
                            <span
                              className={`text-xl font-semibold $ text-green-600`}
                            >
                              {toDollar(calc.getProfit())}
                            </span>
                            <span className="inline"> last sale's profits</span>
                          </div>
                        )}

                        <InfoTooltip side="right">
                          This is the profit you have made from the last sale of
                          this stock. It is calculated with the amount spent on
                          your remaining shares and <strong>last</strong> sale
                          amount. If you never sold, this value will remain
                          zero.
                        </InfoTooltip>
                      </div>
                    </div>
                    {thisStock?.current_price && (
                      <>
                        <div className="text-xs font-medium text-gray-600 dark:text-gray-300 inline">
                          <div className="flex items-center gap-1">
                            <span className="text-xl animate-pulse font-semibold text-gray-900 dark:text-white">
                              {toDollar(
                                calc.getTotalValue(thisStock.current_price)
                              )}
                            </span>
                            <span className="inline">current value</span>
                            <InfoTooltip side="right">
                              This is the current value of your shares based on
                              the current price of the stock.
                            </InfoTooltip>
                          </div>
                        </div>
                        {calc.getTotalShares() > 0 && (
                          <div className="text-xs font-medium text-gray-600 dark:text-gray-300 inline">
                            {calc.getTotalProfit(thisStock.current_price) <
                            0 ? (
                              <div className="flex items-center gap-1">
                                <span
                                  className={`text-xl animate-pulse font-semibold text-red-600`}
                                >
                                  {toDollar(
                                    calc.getTotalProfit(
                                      thisStock.current_price
                                    ) * -1
                                  )}
                                </span>
                                <span className="inline">potential loss</span>
                                <InfoTooltip side="right">
                                  This is the potential loss you would make if
                                  you sold all your shares at the current price.
                                </InfoTooltip>
                              </div>
                            ) : (
                              <div className="flex items-center gap-1">
                                <span
                                  className={`text-xl animate-pulse font-semibold text-green-600`}
                                >
                                  {toDollar(
                                    calc.getTotalProfit(thisStock.current_price)
                                  )}
                                </span>
                                <span className="inline">potential profit</span>
                                <InfoTooltip side="right">
                                  This is the potential profit you would make if
                                  you sold all your shares at the current price.
                                </InfoTooltip>
                              </div>
                            )}
                          </div>
                        )}
                        <p className="text-[10px] font-light">
                          Based on prices from{" "}
                          {moment(thisStock?.timestamp).fromNow()}
                        </p>
                      </>
                    )}
                  </div>
                </>
              ) : (
                <p className="text-sm font-medium text-gray-600 dark:text-gray-300">
                  No purchase history
                </p>
              )}
              <div className="flex flex-col md:items-center mt-2 gap-1 w-full">
                <Button
                  variant="secondary"
                  className="hover:invert flex items-center"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    navigate(`/stocks?ticker=${ticker}`);
                  }}
                >
                  <Pencil />
                  Edit
                </Button>
                <Button
                  variant="secondary"
                  style={{
                    backgroundColor: colors[0],
                    color: colors[1] ?? "white",
                  }}
                  className={`flex items-center hover:bg-gray-900 hover:text-white dark:hover:bg-gray-300 dark:hover:text-black`}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    navigate(`/stocks/${ticker}`);
                  }}
                >
                  Visit
                  <ArrowRight />
                </Button>
              </div>
            </>
          )}
        </div>
      </div>
    </span>
  );
}
