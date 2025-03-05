import { useSupabase } from "@/database/SupabaseProvider";
import Stock_Chart from "@/components/stock_chart_demo";
// import {
//   IoMdInformationCircleOutline,
//   IoMdInformationCircle,
// } from "react-icons/io";
// import {
//   HoverCard,
//   HoverCardContent,
//   HoverCardTrigger,
// } from "@/components/ui/hover-card";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useEffect } from "react";
import { MdEdit } from "react-icons/md";
import useAsync from "@/hooks/useAsync";
import { toast } from "sonner";
import { type Stock } from "@/types/stocks";
import { useQuery } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import RadialChart from "@/components/radial-chart";
import { GenerateStockLLM } from "@/components/llm/stock-llm";
import { cache_keys } from "@/lib/constants";
import Predictions from "@/components/predictions";
import PredictionTable from "@/components/ui/prediction-table";
import { Separator } from "@/components/ui/separator";

const staticStockData = [
  { stock_ticker: "TSLA", stock_name: "Tesla" },
  { stock_ticker: "F", stock_name: "Ford" },
  { stock_ticker: "GM", stock_name: "General Motors" },
  { stock_ticker: "TM", stock_name: "Toyota Motor Corporation" },
  { stock_ticker: "STLA", stock_name: "Stellantis N.V." },
];

interface StockResponse {
  Stocks: {
    stock_id: number;
    stock_name: string;
    stock_ticker: string;
  };
  shares_owned: number;
  desired_investiture: number;
}

export default function Stocks() {
  const { displayName, supabase, user } = useSupabase();
  const { ticker }: { ticker?: string } = useParams();
  const navigate = useNavigate();
  const { data: stocksFetch, error: availableStocksError } = useQuery<Stock[]>({
    queryKey: [cache_keys.USER_STOCKS, ticker],
    queryFn: async () => {
      const { data, error } = await supabase.from("Stocks").select("*");
      if (error) throw error;
      return data || [];
    },
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

  const { value: stocks, error: stocksError } = useAsync<StockResponse[]>(
    () =>
      new Promise((resolve, reject) => {
        supabase
          .from("User_Stocks")
          .select("Stocks (*), shares_owned, desired_investiture")
          .eq("user_id", user?.id)
          .order("created_at", { ascending: false })
          .limit(5)
          .then(({ data, error }) => {
            if (error) reject(error);
            // @ts-expect-error Stocks will never expand to an array
            resolve(data || []);
          });
      }),
    [user, supabase]
  );
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

  if (stocksError) {
    return (
      <div className="flex flex-col justify-center items-center h-screen">
        <h1 className="text-3xl">Error</h1>
        <p className="text-primary">
          Unfortunately, we encountered an error fetching your stocks. Please
          refresh the page or try again later.
        </p>
      </div>
    );
  }

  const hype_meter = 0.365913391113281;
  const currentStock = stocks?.find(
    (stock) =>
      stock?.Stocks?.stock_name ===
      ticker_name?.[ticker as keyof typeof ticker_name]
  );
  return (
    <div className="lg:p-4 md:w-10/12 w-full mx-auto">
      <h1 className="font-semibold text-3xl pb-6">
        {ticker_name
          ? ticker_name[ticker as keyof typeof ticker_name] || "Undefined"
          : "Stock not found"}
      </h1>
      <GenerateStockLLM ticker={ticker} />
      <div className="border border-black dark:border-white p-4 bg-secondary dark:bg-dark rounded-md w-full">
        <div className="relative">
          <Link to="/stocks">
            <MdEdit className="absolute right-0 top-1/2 transform -translate-y-1/2 transition-transform duration-300 hover:scale-125" />
          </Link>
        </div>
        <h2 className="font-semibold md:text-lg text-xs">Hey {displayName},</h2>
        <h3 className="md:text-md text-xs">Current Stock Rate: $ 10.12</h3>
        <h3 className="md:text-md text-xs">
          Money Available to Invest: ${" "}
          {stocks?.find(
            (stock) =>
              stock?.Stocks?.stock_name ===
              ticker_name?.[ticker as keyof typeof ticker_name]
          )?.desired_investiture ?? "N/A"}
        </h3>
        <div className="flex md:flex-row flex-col justify-center lg:gap-64 md:gap-32 gap:5 mt-4">
          <div className="flex flex-col">
            <h3 className="lg:text-lg text-md">Number of Stocks Invested:</h3>
            <p className="lg:text-4xl md:text-3xl text-2xl">
              {stocks?.find(
                (stock) =>
                  stock?.Stocks?.stock_name ===
                  ticker_name?.[ticker as keyof typeof ticker_name]
              )?.shares_owned ?? "N/A"}
            </p>
          </div>
          <div className="flex flex-col">
            <h3 className="lg:text-lg text-md">Current Stock Earnings:</h3>
            <p className="lg:text-4xl md:text-3xl text-2xl">$101.12</p>
          </div>
        </div>
      </div>

      <div className="flex flex-col md:items-center pt-4">
        <Stock_Chart ticker={ticker ?? ""} />
      </div>
      <div className="flex flex-col md:items-center pt-4">
        <Card className="border border-black dark:border-white w-full p-1">
          <CardTitle className="font-semibold text-3xl my-2">
            Predictions
            <Separator />
          </CardTitle>
          <CardContent>
            <div className="grid grid-cols-6 gap-2">
              <div className="col-span-6 lg:col-span-2">
                <h2 className="font-semibold text-lg">Suggestion</h2>
              </div>
              <div className="col-span-6 lg:col-span-4">
                {currentStock && (
                  <PredictionTable ticker={currentStock?.Stocks.stock_ticker} />
                )}
              </div>
            </div>
            {currentStock && <Predictions {...currentStock?.Stocks} />}
          </CardContent>
        </Card>
      </div>
      <div className="flex flex-col md:items-center gap-4 mt-4 w-full">
        <div className="grid grid-cols-6 gap-2">
          <div className="col-span-6 xl:col-span-3">
            <Card className="border border-black dark:border-white rounded-md md:p-4">
              <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
                <div className="grid flex-1 gap-1 sm:text-left">
                  <CardTitle className="text-center font-semibold text-md md:text-lg lg:text-xl">
                    Hype Meter
                  </CardTitle>
                  <CardDescription>
                    Hype Meter analyzes social media sentiment to forecast stock
                    market trends.
                  </CardDescription>
                </div>
              </CardHeader>
              <div className="flex flex-col md:flex-row items-center justify-center">
                <RadialChart score={hype_meter} />
              </div>
            </Card>
          </div>
          <div className="col-span-6 xl:col-span-3">
            <Card className="border border-black dark:border-white rounded-md md:p-4">
              <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
                <div className="grid flex-1 gap-1 sm:text-left">
                  <CardTitle className="text-center font-semibold text-md md:text-lg lg:text-xl">
                    Impact Factor
                  </CardTitle>
                  <CardDescription>
                    Impact Factor scores how major events like elections,
                    natural disasters, and regulations influence stock
                    performance
                  </CardDescription>
                </div>
              </CardHeader>
              <div className="flex flex-col md:flex-row items-center justify-center">
                <RadialChart score={hype_meter} />
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
