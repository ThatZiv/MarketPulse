import { useState } from "react";
import { useSupabase } from "@/database/SupabaseProvider";
import Stock_Chart from "@/components/stock_chart_demo";
import TimeFrameDropdownMenu from "@/components/time-frame-dropdown";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useEffect } from "react";
import { AiOutlineRise } from "react-icons/ai";
import GaugeComponent from "react-gauge-component";
import useAsync from "@/hooks/useAsync";
import { toast } from "sonner";
import { type Stock } from "@/types/stocks";
import { useQuery } from "@tanstack/react-query";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
  CardContent,
} from "@/components/ui/card";
import RadialChart from "@/components/radial-chart";
import { GenerateStockLLM } from "@/components/llm/stock-llm";
import { Button } from "@/components/ui/button";
import { Pie_Chart } from "@/components/pie-chart";
import CustomTable from "@/components/custom-table";
import PredictionChart from "@/components/prediction-chart";
import TransactionHistory from "@/components/history-accordion";
const staticStockData = [
  { stock_ticker: "TSLA", stock_name: "Tesla" },
  { stock_ticker: "F", stock_name: "Ford" },
  { stock_ticker: "GM", stock_name: "General Motors" },
  { stock_ticker: "TM", stock_name: "Toyota Motor Corporation" },
  { stock_ticker: "STLA", stock_name: "Stellantis N.V." },
];

const meters = [
  {
    "Hype Meter":
      "Hype Meter analyzes social media sentiment to forecast stock market trends.",
  },
  {
    "Disruption Score":
      "Disruption Score measures the potential impact on stock prices from supply chain delays or shifts.",
  },
  {
    "Impact Factor":
      "Impact Factor scores how major events like elections, natural disasters, and regulations influence stock performance.",
  },
];
interface StockResponse {
  Stocks: {
    stock_name: string;
  };
  shares_owned: number;
  desired_investiture: number;
}

const sample_predictions = [
  {
    date: '2025-01-02',
    predicted_price: 200
  },
  {
    date: '2025-01-03',
    predicted_price: 220
  },
  {
    date: '2025-01-06',
    predicted_price: 212
  },
  {
    date: '2025-01-07',
    predicted_price: 202
  },
  {
    date: '2025-01-08',
    predicted_price: 202
  },
  {
    date: '2025-01-09',
    predicted_price: 213
  },
  {
    date: '2025-01-13',
    predicted_price: 216
  }
]

export default function Stocks() {
  const { displayName, supabase, user } = useSupabase();
  const { ticker }: { ticker?: string } = useParams();
  const options = ["One Day", "One Week", "One Month"]
  const [timeFrameSelected, setTimeFrameSelected] = useState(options[0]);
  const navigate = useNavigate();
  const { data: stocksFetch, error: availableStocksError } = useQuery<Stock[]>({
    queryKey: ["stocksFetch"],
    queryFn: async () => {
      const { data, error } = await supabase.from("Stocks").select("*");
      if (error) throw error;
      return data || [];
    },
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
          .select("Stocks (stock_name), shares_owned, desired_investiture")
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
      console.log(ticker_name?.[ticker as keyof typeof ticker_name]);
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
  const impact_factor = 10;
  const disruption_score = 40;
  const hype_meter = 0.365913391113281;
  return (
    <div className="lg:p-4 md:w-10/12 w-xl mx-auto">
      <h1 className="font-semibold text-3xl pb-6">
        {ticker_name
          ? ticker_name[ticker as keyof typeof ticker_name] || "Undefined"
          : "Stock not found"}
      </h1>
      <GenerateStockLLM ticker={ticker} />
      <div className="flex justify-end right-0 gap-4 py-2">
        <Button variant="edit" asChild>
          <Link to="/stocks">Edit</Link>
        </Button>
        <Button variant='delete'>Delete</Button>
      </div>
      <div className="border border-black dark:border-white p-4 bg-secondary dark:bg-dark rounded-md w-full">

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
          <div className="flex flex-col">
            <h3 className="lg:text-lg text-md pb-6">Today's gains:</h3>
            <div className="flex flex-row lg:text-4xl md:text-3xl text-2xl">
              <h3 className="lg:text-4xl md:text-3xl text-2xl">$24.45</h3>
              <AiOutlineRise />
            </div>
          </div>
        </div>
      </div>

      <div className="flex flex-col md:items-center pt-4">
        {/* <Stock_Chart ticker={ticker ?? ""} /> */}
      </div>
      <Card className="flex flex-col p-2 my-3 border border-black dark:border-white dark:bg-dark rounded-md w-full">
        <CardHeader>
          <CardDescription className="text-2xl">Recommendations:</CardDescription>
          <CardContent className="flex flex-col md:items-center pt-4">
            <div className="flex flex-row justify-center lg:gap-14 mb-4">
              <Pie_Chart />
              <div className="flex flex-col justify-start w-full">
                <h3 className="flex flex-row sm:text-md lg:text-lg font-semibold gap-4">
                  Prediction Time Frame:
                  <TimeFrameDropdownMenu values={options} selectedValue={timeFrameSelected} onChange={setTimeFrameSelected} />
                </h3>
                <CustomTable caption={"Predicted prices"} tableheader={["Prediction", "Date"]} predictions={sample_predictions} />
              </div>
            </div>
            <PredictionChart ticker={ticker ?? ""} predictions={sample_predictions} />
          </CardContent>
        </CardHeader>
      </Card>
      <Card className="p-4">
        <TransactionHistory ticker={ticker??""}/>
      </Card>
      <div className="flex flex-col md:items-center gap-4 mt-4 w-full">
        <Card className="border border-black dark:border-white rounded-md md:p-4">
          <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
            <div className="grid flex-1 gap-1 sm:text-left">
              <CardTitle className="text-center font-semibold text-md md:text-lg lg:text-xl">
                {Object.keys(meters[0])[0]}
              </CardTitle>
              <CardDescription>{meters[0]["Hype Meter"]}</CardDescription>
            </div>
          </CardHeader>
          <div className="flex flex-col md:flex-row items-center justify-center gap-5">
            <RadialChart score={hype_meter} />
          </div>
        </Card>

        <div className="flex flex-col md:flex-row justify-between gap-4 md:mt-4 md:max-w-9/12 lg:max-w-full max-w-full">
          <Card className="flex flex-col items-center justify-between border border-black dark:border-white md:w-1/2 rounded-md">
            <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
              <div className="grid flex-1 gap-1 sm:text-left">
                <CardTitle className="text-center font-semibold text-md md:text-lg lg:text-xl">
                  {Object.keys(meters[1])[0]}
                </CardTitle>
                <CardDescription>
                  {meters[1]["Disruption Score"]}
                </CardDescription>
              </div>
            </CardHeader>
            <div className="lg:w-60 md:w-full w-96 h-full">
              <GaugeComponent
                style={{ width: "100%", height: "100%" }}
                value={disruption_score}
                type={"radial"}
                labels={{
                  valueLabel: {
                    style: { fill: "var(--tick-label-color)" },
                  },
                  tickLabels: {
                    type: "outer",
                    ticks: [
                      { value: 20 },
                      { value: 40 },
                      { value: 60 },
                      { value: 80 },
                      { value: 100 },
                    ],

                    defaultTickValueConfig: {
                      style: { fill: "var(--tick-label-color)" },
                    },
                  },
                }}
                arc={{
                  colorArray: ["#5BE12C", "#EA4228"],
                  subArcs: [{ limit: 20 }, {}, {}, {}, {}],
                  padding: 0.02,
                  width: 0.4,
                }}
                pointer={{
                  elastic: true,
                  animationDelay: 0,
                  color: "#000000",
                }}
              />
            </div>
          </Card>
          <Card className="flex flex-col items-center justify-between border border-black dark:border-white md:w-1/2 rounded-md">
            <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
              <div className="grid flex-1 gap-1 sm:text-left">
                <CardTitle className="text-center font-semibold text-md md:text-lg lg:text-xl">
                  {Object.keys(meters[2])[0]}
                </CardTitle>
                <CardDescription>{meters[2]["Impact Factor"]}</CardDescription>
              </div>
            </CardHeader>
            <div className="lg:w-60 md:w-full w-96 h-full">
              <GaugeComponent
                style={{ width: "100%", height: "100%" }}
                value={impact_factor}
                type={"radial"}
                labels={{
                  valueLabel: {
                    style: { fill: "var(--tick-label-color)" },
                  },
                  tickLabels: {
                    type: "outer",
                    ticks: [
                      { value: 20 },
                      { value: 40 },
                      { value: 60 },
                      { value: 80 },
                      { value: 100 },
                    ],

                    defaultTickValueConfig: {
                      style: { fill: "var(--tick-label-color)" },
                    },
                  },
                }}
                arc={{
                  colorArray: ["#5BE12C", "#EA4228"],
                  subArcs: [{ limit: 20 }, {}, {}, {}, {}],
                  padding: 0.02,
                  width: 0.4,
                }}
                pointer={{
                  elastic: true,
                  animationDelay: 0,
                  color: "#000000",
                }}
              />
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
