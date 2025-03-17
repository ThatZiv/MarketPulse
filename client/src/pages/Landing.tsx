import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useSupabase } from "@/database/SupabaseProvider";
import { useApi } from "@/lib/ApiProvider";
import { useQueries, useQuery } from "@tanstack/react-query";
import { extractColors } from "extract-colors";
import { Link } from "react-router";
import { cache_keys } from "@/lib/constants";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { LiaSortSolid } from "react-icons/lia";

interface StockResponse {
  Stocks: {
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
  const [sort, setSort] = useState("None");
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
          .select("Stocks (stock_name, stock_ticker), shares_owned")
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
  let sortedStocks: StockResponse[] = [];
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
    }
  }


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
              <div className="flex flex-col gap-6">
                <div className="flex flex-col justify-end items-end gap-6">
                  <div className="flex flex-col justify-center items-start">
                    <div className="flex items-center">
                      <LiaSortSolid className="ml-2" /> 
                      <h3 className="">Sort:</h3>
                    </div>
                    <Select value={sort} onValueChange={setSort}>
                      <SelectTrigger
                        className="md:w-[160px] rounded-lg sm:ml-auto dark:border-white w-[1rem]"
                      >
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
                      </SelectContent>
                    </Select>
                  </div>

                </div>


                <div className="flex flex-row flex-wrap items-center justify-center gap-6">
                  {sortedStocks?.map((stock, index) => (
                    <StockCard
                      key={stock?.Stocks?.stock_name}
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
function StockCard({
  stock,
  img,
  colors,
}: StockCardProps & { img: string; colors: string[] }) {
  return (
    <Link to={`/stocks/${stock.Stocks.stock_ticker}`}>
      <HoverCard>
        <HoverCardTrigger>

          <div
            className="bg-white dark:bg-black p-6 rounded-xl shadow-lg hover:shadow-2xl transition-all transform hover:scale-105 duration-200 ease-in-out"
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
              {stock.Stocks.stock_ticker}
            </h3>

            <Separator className="mb-4  border-2 dark:border-gray-300 border-gray-800" />

            <p className="text-sm font-medium text-gray-600 dark:text-gray-300">
              <span className="text-xl font-semibold text-gray-900 dark:text-white">
                {stock.shares_owned.toLocaleString()}
              </span>{" "}
              shares owned
            </p>
          </div>
        </HoverCardTrigger>
        <HoverCardContent className="w-auto rounded-lg p-2.5">
          {stock.Stocks.stock_name}
        </HoverCardContent>
      </HoverCard>

    </Link>
  );
}
