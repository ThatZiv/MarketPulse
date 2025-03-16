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
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { ArrowRight, Pencil } from "lucide-react";

interface StockResponse {
  Stocks: {
    stock_id: number;
    stock_name: string;
    stock_ticker: string;
  };
  shares_owned: number;
}

interface StockCardProps {
  stock: StockResponse;
}

export default function Landing() {
  const { supabase, displayName, user } = useSupabase();
  const api = useApi();
  const { dispatch } = useGlobal();

  const {
    data: stocks,
    error: stocksError,
    status: stocksStatus,
  } = useQuery<StockResponse[]>({
    queryKey: [cache_keys.USER_STOCKS],
    queryFn: () =>
      new Promise((resolve, reject) => {
        supabase
          .from("User_Stocks")
          .select("Stocks (*), shares_owned")
          .eq("user_id", user?.id)
          .order("created_at", { ascending: false })
          .limit(5)
          .then(({ data, error }) => {
            if (error) reject(error);
            // @ts-expect-error Stocks will never expand to an array
            resolve(data || []);
          });
      }),
  });

  // coroutine that gets the user's stock transactions to global state
  useQuery({
    queryKey: [cache_keys.USER_STOCK_TRANSACTION, "global"],
    queryFn: () =>
      new Promise((resolve, reject) => {
        supabase
          .from("User_Stock_Purchases")
          .select("Stocks (stock_ticker), *")
          .eq("user_id", user?.id)
          .order("date", { ascending: false })
          .then(({ data, error }) => {
            if (error) {
              reject(error);
              return;
            }
            const allTransactions: {
              [ticker: string]: Array<
                PurchaseHistoryDatapoint & { Stocks: { stock_ticker: string } }
              >;
            } = {};
            data?.forEach((transaction) => {
              if (!allTransactions[transaction.Stocks.stock_ticker]) {
                allTransactions[transaction.Stocks.stock_ticker] = [];
              }
              allTransactions[transaction.Stocks.stock_ticker].push(
                transaction
              );
            });
            Object.entries(allTransactions).forEach(
              ([ticker, transactions]) => {
                dispatch({
                  type: actions.SET_USER_STOCK_TRANSACTIONS,
                  payload: { data: transactions, stock_ticker: ticker },
                });
              }
            );
            resolve(data || []);
          });
      }),
  });

  const stockImages = useQueries({
    queries:
      stocks?.map((stock) => ({
        queryKey: ["stock", stock.Stocks.stock_ticker],
        queryFn: () => api?.getStockLogo(stock.Stocks.stock_ticker),
        staleTime: Infinity,
      })) || [],
  }).map((query) => query.data);

  const stockColors = useQueries({
    queries:
      stockImages?.map((img) => ({
        queryKey: ["stock", img],
        queryFn: () => extractColors(img ?? ""),
        staleTime: Infinity,
      })) || [],
  })
    ?.map((query) => query.data)
    .map(
      // sort by most common color first
      (img) => img?.sort((a, b) => b.area - a.area).map((color) => color.hex)
    );

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
    <div className="min-h-screen">
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
              <div className="flex flex-row flex-wrap items-center justify-center gap-6">
                {stocks?.map((stock, index) => (
                  <StockCard
                    key={stock?.Stocks?.stock_name}
                    stock={stock}
                    img={stockImages[index] ?? ""}
                    colors={stockColors[index] ?? []}
                  />
                ))}
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
function StockCard({
  stock,
  img,
  colors,
}: StockCardProps & { img: string; colors: string[] }) {
  const {
    state: { history },
  } = useGlobal();
  const navigate = useNavigate();
  const ticker = stock.Stocks.stock_ticker;
  const userStockHistory = history[ticker];
  const calc = useMemo(
    () => new PurchaseHistoryCalculator(userStockHistory ?? []),
    [userStockHistory]
  );
  const [hovered, setHovered] = useState(false);

  return (
    <Link
      to={`/stocks/${ticker}`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div
        className="bg-white hover:bg-slate-200 hover:p-8 dark:hover:bg-gray-800 dark:bg-black p-6 rounded-xl shadow-lg hover:shadow-2xl transition-all hover:scale-105 transform duration-500 ease-in-out"
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

        {/* Transition for the hovered content */}
        <div
          className={`flex flex-col items-center overflow-hidden transition-all duration-700 ease-in-out ${
            hovered ? "max-h-[1000px] opacity-100" : "max-h-0 opacity-0"
          }`}
        >
          {hovered && (
            <>
              {" "}
              {userStockHistory ? (
                <>
                  <Separator className="mb-4 border-2 dark:border-gray-300 border-gray-800" />
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {stock.Stocks.stock_name}
                  </p>
                  <div className="flex flex-col items-center w-full">
                    <p className="text-sm font-medium text-gray-600 dark:text-gray-300">
                      <span className="text-xl font-semibold text-gray-900 dark:text-white">
                        {calc.getTotalShares()}
                      </span>{" "}
                      share{calc.getTotalShares() === 1 ? "" : "s"} owned
                    </p>
                    <p className="text-sm font-medium text-gray-600 dark:text-gray-300">
                      <span className="text-xl font-semibold text-gray-900 dark:text-white">
                        {PurchaseHistoryCalculator.toDollar(
                          calc.getTotalBought()
                        )}
                      </span>{" "}
                      purchased
                    </p>
                  </div>
                </>
              ) : (
                <p className="text-sm font-medium text-gray-600 dark:text-gray-300">
                  No purchase history
                </p>
              )}
              <div className="flex items-center mt-2 gap-1">
                <Button
                  variant="secondary"
                  className="hover:invert flex items-center"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    navigate(`/stocks/?ticker=${ticker}`);
                  }}
                >
                  <Pencil />
                  Edit
                </Button>
                <Button
                  variant="default"
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
    </Link>
  );
}
