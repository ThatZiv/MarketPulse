import { useSupabase } from "@/database/SupabaseProvider";
import HistoricalChart from "@/components/historical-chart";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import { MdEdit } from "react-icons/md";
import { toast } from "sonner";
import { useQuery } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { GenerateStockLLM } from "@/components/llm/stock-llm";
import { cache_keys } from "@/lib/constants";
import Predictions from "@/components/predictions";
import PredictionTable from "@/components/ui/prediction-table";
import { Separator } from "@/components/ui/separator";
import { useGlobal } from "@/lib/GlobalProvider";
import { StocksState } from "@/types/global_state";
import moment from "moment";
import Recommendation from "@/components/recommendation-chart";
import { Button } from "@/components/ui/button";
import PurchaseHistory from "@/components/purchase-history";
import { SentimentMeter } from "@/components/sentiment-meter";
import { Skeleton } from "@/components/ui/skeleton";
import { PurchaseHistoryCalculator } from "@/lib/Calculator";
import { DeleteStock } from "@/components/delete-stock";
import dataHandler from "@/lib/dataHandler";

const staticStockData = [
  { stock_ticker: "TSLA", stock_name: "Tesla", stock_id: 1 },
  { stock_ticker: "F", stock_name: "Ford", stock_id: 2 },
  { stock_ticker: "GM", stock_name: "General Motors", stock_id: 3 },
  { stock_ticker: "TM", stock_name: "Toyota Motor Corporation", stock_id: 4 },
  { stock_ticker: "STLA", stock_name: "Stellantis N.V.", stock_id: 5 },
];

const sentimentFooter = (score: number, meter: string): string => {
  let trend = "";
  if (meter == "hype") {
    trend = "social media";
  } else if (meter == "impact") {
    trend = "news";
  }

  if (score >= 0 && score <= 15) {
    return `Strongly Negative sentiment in ${trend} around the stock`;
  } else if (score > 15 && score <= 30) {
    return `Negative sentiment in ${trend} around the stock`;
  } else if (score > 30 && score <= 45) {
    return `Slightly Negative sentiment in ${trend} around the stock`;
  } else if (score > 45 && score <= 55) {
    return `Neutral sentiment in ${trend} around the stock`;
  } else if (score > 55 && score <= 70) {
    return `Slightly Positive sentiment in ${trend} around the stock`;
  } else if (score > 70 && score <= 85) {
    return `Positive sentiment in ${trend} around the stock`;
  } else if (score > 85 && score <= 100) {
    return `Strongly Positive sentiment in ${trend} around the stock`;
  } else {
    return "N/A";
  }
};
export default function Stocks() {
  const { supabase, user } = useSupabase();
  const { ticker }: { ticker?: string } = useParams();
  const { state } = useGlobal();
  const navigate = useNavigate();
  const [hype_meter, setHypeMeter] = useState(0);
  const [impact_meter, setImpactMeter] = useState(0);
  const { data: stocksFetch, error: availableStocksError } = useQuery({
    queryKey: [cache_keys.STOCKS],
    queryFn: dataHandler().forSupabase(supabase).getAllStocks(),
    enabled: !!ticker,
  });

  const availableStocks =
    stocksFetch?.map((stock) => ({
      [stock.stock_ticker]: stock.stock_name,
    })) ||
    staticStockData.map((stock) => ({
      [stock.stock_ticker]: stock.stock_name,
    }));

  const ticker_name = availableStocks.find(
    (stock) => stock[ticker as keyof typeof stock]
  );

  const { data: stocks, error: stocksError } = useQuery({
    queryKey: [cache_keys.USER_STOCKS],
    staleTime: Infinity,
    queryFn: dataHandler()
      .forSupabase(supabase)
      .getUserStocks(user?.id ?? ""),
    enabled: !!user?.id,
  });

  const meters = useMemo(() => {
    // gets the last data point in stock data for a given ticker
    const defaultMeter = {
      date: new Date("bad date"),
      value: 0,
      state: "loading",
    };
    if (!ticker || !state.stocks[ticker]) {
      //def a better way to do this
      return {
        hype: defaultMeter,
        impact: defaultMeter,
      };
    }
    const getLastStockHistory = (
      stockHistory: StocksState["history"],
      key: string
    ) => {
      if (!stockHistory) {
        return defaultMeter;
      }
      const t = stockHistory.sort(
        (a, b) =>
          new Date(a.time_stamp.join(" ")).getTime() -
          new Date(b.time_stamp.join(" ")).getTime()
      );
      const last = t[t.length - 1];
      return {
        date: new Date((last?.time_stamp ?? []).join(" ")),
        value: (last?.[key as keyof typeof last] as number) ?? 0,
      };
    };
    const hype_meter = getLastStockHistory(
      state.stocks[ticker ?? ""].history,
      "hype_meter"
    );
    const impact_meter = getLastStockHistory(
      state.stocks[ticker ?? ""].history,
      "impact_factor"
    );
    return {
      hype: hype_meter,
      impact: impact_meter,
    };
  }, [state.stocks, ticker]);
  useEffect(() => {
    if (!ticker_name) {
      // Redirect
      navigate("/");
      toast.error(
        "Invalid ticker: The entered ticker is not found in our database."
      );
      return;
    }
  });
  useEffect(() => {
    if (availableStocksError) {
      toast.error(
        `Error: ${availableStocksError.message || "An unknown error occurred"}`
      );
    }
  }, [availableStocksError]);
  useEffect(() => {
    if (!stocks || stocks.length === 0) {
      return;
    }
    const tickerToCheck = ticker_name?.[ticker as keyof typeof ticker_name];
    const stockExists = stocks?.some(
      (stock) => stock?.Stocks?.stock_name === tickerToCheck
    );
    if (!stockExists) {
      navigate("/");
      toast.warning(
        "Restricted access: To view this page, please add this ticker to your account."
      );
      return;
    }
  }, [stocks]);
  useEffect(() => {
    if (meters.hype.value) {
      const hype_temp = ((meters.hype.value + 6) / 12) * 100;
      setHypeMeter(hype_temp);
    }
    if (meters.impact.value || meters.impact.value == 0) {
      const impact_temp = ((meters.impact.value + 7.5) / 15.9) * 100;
      setImpactMeter(impact_temp);
    }
  }, [meters.hype.value, meters.impact.value]);

  const calc = useMemo(
    () =>
      new PurchaseHistoryCalculator(
        ticker ? (state.history[ticker] ?? []) : []
      ),
    [state.history, ticker]
  );

  if (stocksError) {
    return (
      <div className="flex flex-col justify-center items-center h-screen">
        <h1 className="text-3xl">Error</h1>
        <p className="text-gray-600">
          Unfortunately, we encountered an error fetching your stocks. Please
          refresh the page or try again later.
        </p>
      </div>
    );
  }

  const currentStock = stocks?.find(
    (stock) =>
      stock?.Stocks?.stock_name ===
      ticker_name?.[ticker as keyof typeof ticker_name]
  );
  return (
    <div className="md:w-10/12 w-full mx-auto">
      <h1 className="font-semibold text-3xl pb-6">
        {ticker_name
          ? ticker_name[ticker as keyof typeof ticker_name] || "Undefined"
          : "Stock not found"}
      </h1>
      <GenerateStockLLM ticker={ticker} />
      <div className="border border-black dark:border-white p-4 bg-secondary dark:bg-dark rounded-md w-full">
        <div className="flex justify-end right-0 gap-4 py-2">
          <Link to={`/stocks?ticker=${ticker}`}>
            <Button variant="secondary" size="sm">
              <MdEdit className="" /> Edit
            </Button>
          </Link>
          <DeleteStock
            stock_id={currentStock?.Stocks.stock_id}
            ticker={
              ticker_name
                ? ticker_name[ticker as keyof typeof ticker_name]
                : undefined
            }
          />
        </div>
        <div className="flex md:flex-row flex-col justify-center lg:gap-64 md:gap-32 gap:5 mt-4">
          <div className="flex flex-col">
            <h3 className="lg:text-2xl text-md">Shares Owned</h3>
            <Separator />
            <p className="lg:text-4xl md:text-3xl text-2xl">
              {calc.getTotalShares()}{" "}
            </p>
          </div>
          <div className="flex flex-col">
            <h3 className="lg:text-2xl text-md">Current Price</h3>
            <Separator />
            <p className="lg:text-4xl md:text-3xl text-2xl">
              ${state.stocks[ticker ?? ""]?.current_price?.toFixed(2) ?? "N/A"}
            </p>
            <p className="text-xs italic">
              Last updated{" "}
              {moment(state.stocks[ticker ?? ""]?.timestamp).fromNow()}
            </p>
          </div>
        </div>
      </div>
      <div className="flex flex-col md:items-center pt-4">
        {currentStock && (
          <HistoricalChart
            ticker={ticker ?? ""}
            stock_id={currentStock?.Stocks.stock_id}
          />
        )}
      </div>
      <div className="flex flex-col md:items-center pt-4">
        <Card className="border border-black dark:border-white w-full p-1">
          <CardTitle className="font-semibold text-center md:text-left text-3xl mx-6 my-5">
            Predictions
          </CardTitle>
          <Separator className="my-3" />
          <CardContent>
            <div className="mb-3">
              {currentStock && <Predictions {...currentStock?.Stocks} />}
              <Separator className="my-3" />
            </div>
            <div className="grid grid-cols-6 gap-2">
              <div className="col-span-6 lg:col-span-2">
                {currentStock && ticker && (
                  <Recommendation stock_ticker={ticker} />
                )}
              </div>
              <div className="col-span-6 lg:col-span-4">
                {currentStock && (
                  <PredictionTable ticker={currentStock?.Stocks.stock_ticker} />
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
      <div className="flex flex-col md:items-center gap-4 mt-4 w-full">
        <div className="grid grid-cols-6 gap-2">
          <div className="col-span-6 xl:col-span-3">
            {moment(meters.hype.date).isValid() ? (
              <Card className="border border-black dark:border-white rounded-md md:p-4 overflow-x-auto h-full">
                <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
                  <div className="grid flex-1 gap-1 sm:text-left">
                    <CardTitle className="text-center font-semibold text-md md:text-lg lg:text-xl">
                      Hype Meter
                    </CardTitle>
                    <CardDescription>
                      <i>Hype Meter</i> analyzes social media sentiment to
                      capture the public's view of a stock. A higher score
                      indicates more positive outlook on the stock among social
                      media users.
                      <Separator className="my-2" />
                      <div className="text-xs">
                        As of {moment(meters.hype.date).calendar()}{" "}
                      </div>
                    </CardDescription>
                  </div>
                </CardHeader>
                <div className="flex flex-col md:flex-row items-center justify-center py-8">
                  <SentimentMeter score={hype_meter} />
                </div>
                <CardFooter className="flex justify-center">
                  <p className="gap-2 font-medium leading-none">
                    {sentimentFooter(hype_meter, "hype")}
                  </p>
                </CardFooter>
              </Card>
            ) : (
              <div className="border bg-white dark:bg-black border-black dark:border-white rounded-md md:p-4 overflow-x-auto h-full">
                <div className="text-center font-semibold text-md md:text-lg lg:text-xl pt-2">
                  Hype Meter
                </div>
                <div className="flex items-center gap-2 space-y-0 border-b py-2 sm:flex-row mb-2">
                  <div className="sm:text-left text-black dark:text-white text-sm text-muted-foreground">
                    <i>Hype Meter</i> analyzes social media sentiment to capture
                    the public's view of a stock. A higher score indicates more
                    positive outlook on the stock among social media users.
                  </div>
                </div>
                <div className="flex flex-col justify-center border border-grey-500 dark:border-white rounded-xl shadow-md p-2">
                  <div className="flex flex-col md:flex-row items-center justify-center py-8">
                    <Skeleton className="w-64 h-32 bg-gray-200 dark:bg-gray-700 rounded-t-full" />
                  </div>
                  <Skeleton className="w-full h-3 my-2 flex justify-center items-center bg-gray-200 dark:bg-gray-700"></Skeleton>
                  <Skeleton className="w-full h-3 my-2 flex justify-center items-center bg-gray-200 dark:bg-gray-700"></Skeleton>
                </div>
                <div className="text-sm text-center text-black dark:text-white p-2">
                  Hype Meter Data currently not available. Please Try Again
                  Later.
                </div>
              </div>
            )}
          </div>
          <div className="col-span-6 xl:col-span-3">
            {moment(meters.hype.date).isValid() ? (
              <Card className="border border-black dark:border-white rounded-md md:p-4 overflow-x-auto h-full">
                <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
                  <div className="grid flex-1 gap-1 sm:text-left">
                    <CardTitle className="text-center font-semibold text-md md:text-lg lg:text-xl">
                      Impact Factor
                    </CardTitle>
                    <CardDescription>
                      <i>Impact Factor</i> scores how major news events like
                      elections, natural disasters, and regulations influence
                      stock performance. A higher score indicates a more
                      positive impact on the stock.
                      <Separator className="my-2" />
                      <div className="text-xs">
                        As of {moment(meters.impact?.date).calendar()}{" "}
                      </div>
                    </CardDescription>
                  </div>
                </CardHeader>
                <div className="flex flex-col md:flex-row items-center justify-center py-8">
                  <SentimentMeter score={impact_meter} />
                </div>
                <CardFooter className="flex justify-center">
                  <p className="gap-2 font-medium leading-none">
                    {sentimentFooter(impact_meter, "impact")}
                  </p>
                </CardFooter>
              </Card>
            ) : (
              <div className="border bg-white dark:bg-black border-black dark:border-white rounded-md md:p-4 overflow-x-auto h-full">
                <div className="text-center font-semibold text-md md:text-lg lg:text-xl pt-2">
                  Impact Factor
                </div>
                <div className="flex items-center gap-2 space-y-0 border-b py-2 sm:flex-row mb-2">
                  <div className="sm:text-left text-black dark:text-white text-sm text-muted-foreground">
                    <i>Impact Factor</i> scores how major news events like
                    elections, natural disasters, and regulations influence
                    stock performance. A higher score indicates a more positive
                    impact on the stock.
                  </div>
                </div>
                <div className="flex flex-col justify-center border border-grey-500 dark:border-white rounded-xl shadow-md p-2">
                  <div className="flex flex-col md:flex-row items-center justify-center py-8">
                    <Skeleton className="w-64 h-32 bg-gray-200 dark:bg-gray-700 rounded-t-full" />
                  </div>
                  <Skeleton className="w-full h-3 my-2 flex justify-center items-center bg-gray-200 dark:bg-gray-700"></Skeleton>
                  <Skeleton className="w-full h-3 my-2 flex justify-center items-center bg-gray-200 dark:bg-gray-700"></Skeleton>
                </div>
                <div className="text-sm text-center text-black dark:text-white p-2">
                  Impact Factor Data currently not available. Please Try Again
                  Later.
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      <div>
        {currentStock && ticker && (
          <PurchaseHistory
            stock_id={currentStock.Stocks.stock_id}
            ticker={ticker}
          />
        )}
      </div>
    </div>
  );
}
