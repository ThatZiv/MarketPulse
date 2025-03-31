import * as React from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Label as ChartLabel,
  XAxis,
  YAxis,
} from "recharts";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useApi } from "@/lib/ApiProvider";
import { useGlobal } from "@/lib/GlobalProvider";
import { useQuery } from "@tanstack/react-query";
import { StockDataItem } from "@/types/stocks";
import { actions, cache_keys } from "@/lib/constants";
import moment from "moment";
import { capitalizeFirstLetter, formatNumber, isSameDay } from "@/lib/utils";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import InfoTooltip from "./InfoTooltip";
interface StockChartProps {
  ticker: string;
  stock_id: number;
}

const normalizeName = (name: string) =>
  name.split("_").map(capitalizeFirstLetter).join(" ");

// FIXME:
// - ChartToolTipContent needs to show date

export default function HistoricalChart({ ticker, stock_id }: StockChartProps) {
  const [timeRange, setTimeRange] = React.useState("7d");
  const [dataInput, setDataInput] =
    React.useState<keyof StockDataItem>("stock_close");
  const api = useApi();
  const { dispatch } = useGlobal();
  const [showPredictions, setShowPredictions] = React.useState(true);

  const timeRangeValue = React.useMemo(() => {
    return parseInt(timeRange.match(/(\d+)/)?.[0] || "0");
  }, [timeRange]);

  // React.useEffect(() => {
  //   queryClient.invalidateQueries({
  //     queryKey: [cache_keys.STOCK_PREDICTION, stock_id],
  //   });
  // }, [timeRangeValue]);

  React.useEffect(() => {
    return () => {
      setShowPredictions(true);
      setDataInput("stock_close");
    };
  }, [ticker]);

  const { data, isLoading, isError } = useQuery<StockDataItem[], Error>({
    queryKey: [cache_keys.STOCK_DATA, ticker],
    queryFn: async () => {
      const data = await api?.getStockData(ticker);
      if (!data) {
        throw new Error("Failed to fetch stock data");
      }
      dispatch({
        type: actions.SET_STOCK_HISTORY,
        payload: { data, stock_ticker: ticker },
      });
      return data.map((item) => ({
        date: new Date(item.time_stamp.join(" ")),
        ...item,
      }));
    },
    enabled: !!api && !!ticker,
  });

  const {
    data: predictionHistory,
    isLoading: arePredictionsLoading,
    error: predictionsError,
  } = useQuery({
    queryKey: [cache_keys.STOCK_PREDICTION, stock_id],
    queryFn: () => api?.getStockPredictions(ticker, 30),
    enabled: !!stock_id && !!api && !!ticker && timeRangeValue > 0,
  });

  const chartData = React.useMemo(() => {
    if (!data) return [];
    if (!predictionHistory) return [];
    // TODO: skip weekends
    const filteredData = data?.map((item) => {
      const thisDate = moment(new Date(item.time_stamp.join(" ")))
        //.add(1, "days") // this will align with prediction forecast
        .toDate();
      const thisPredictionHistory = predictionHistory.find((point) =>
        // isSameDay(thisDate, moment(point.created_at).add(-1, "days").toDate())
        isSameDay(thisDate, moment(point.created_at).toDate())
      );
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const point: Record<string, any> = {
        date: thisDate,
        // stock_close: item.stock_close,
        // stock_open: item.stock_open,
        // stock_high: item.stock_high,
        // stock_low: item.stock_low,
        // stock_volume: item.stock_volume,
      };
      if (dataInput !== "stock_close") {
        setShowPredictions(false);
      }
      point[dataInput] = item[dataInput];
      if (showPredictions) {
        point["prediction"] = thisPredictionHistory?.output.find(
          (point) => point.name === "average"
        )?.forecast[0];
      }
      return point;
    });
    return filteredData.slice(
      filteredData.length - timeRangeValue,
      filteredData.length
    );
  }, [data, timeRangeValue, predictionHistory, showPredictions, dataInput]);

  const chartConfig = React.useMemo<ChartConfig>(() => {
    const config: ChartConfig = {};
    const colors = ["#479BC6", "#ea580c"];
    if (!chartData) return config;
    for (const name of Object.keys(chartData[0] ?? {}) ?? []) {
      if (name === "date") continue;
      config[name] = {
        label: normalizeName(name),
        color: colors.shift() ?? "#606060",
      };
    }
    return config;
  }, [chartData]);

  const YAxisLabels: Record<string, string> = {
    stock_close: "Closing Price ($)",
    stock_open: "Opening Price ($)",
    stock_high: "High Price ($)",
    stock_low: "Low Price ($)",
    stock_volume: "Volume (# shares)",
    sentiment_data: "Social Score - higher is better",
    news_data: "News Score - higher is better",
  };

  return (
    <>
      {isLoading || arePredictionsLoading ? (
        <Card className="w-full p-4 border border-black dark:border-white ">
          {(isLoading || arePredictionsLoading) && (
            <div
              role="status"
              className="border border-gray-200 rounded-sm shadow-sm animate-pulse md:p-6 dark:border-gray-700"
            >
              <div className="">
                <div className="w-48 h-2 my-2 bg-gray-200 rounded-full dark:bg-gray-700"></div>
                <div className="w-32 h-2 my-2 bg-gray-200 rounded-full dark:bg-gray-700"></div>
                <div className="w-52 h-2 my-2 bg-gray-200 rounded-full dark:bg-gray-700"></div>
              </div>
              <div className="flex items-baseline mt-2">
                <div className="w-full bg-gray-200 rounded-t-lg h-72 dark:bg-gray-700"></div>
                <div className="w-full h-56 ms-6 bg-gray-200 rounded-t-lg dark:bg-gray-700"></div>
                <div className="w-full bg-gray-200 rounded-t-lg h-72 ms-6 dark:bg-gray-700"></div>
                <div className="w-full h-64 ms-6 bg-gray-200 rounded-t-lg dark:bg-gray-700"></div>
                <div className="w-full bg-gray-200 rounded-t-lg h-80 ms-6 dark:bg-gray-700"></div>
                <div className="w-full bg-gray-200 rounded-t-lg h-72 ms-6 dark:bg-gray-700"></div>
                <div className="w-full bg-gray-200 rounded-t-lg h-80 ms-6 dark:bg-gray-700"></div>
              </div>
            </div>
          )}
          {(isError || predictionsError) && (
            <div>Error fetching stock data. Please try again later...</div>
          )}
        </Card>
      ) : (
        <Card className="w-full p-4 border border-black dark:border-white">
          <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
            <div className="grid flex-1 gap-1 text-center sm:text-left">
              <CardTitle>{ticker} Historical Prices</CardTitle>
              <CardDescription>
                <div className="text-xs">
                  Updated as of{" "}
                  {new Date(
                    chartData[chartData.length - 1]?.date
                  ).toLocaleDateString()}
                </div>
              </CardDescription>
            </div>
            <div className="flex items-center space-x-2">
              <InfoTooltip side="left">
                <div className="text-xs">
                  Toggle to show or hide the average predictions on the chart.
                  The average predictions are based on the average output from
                  all the forecast models.{" "}
                  <span className="font-bold">
                    You can only view predictions only when viewing stock
                    closing
                  </span>{" "}
                  because the predictions are based on stock closing prices.
                </div>
              </InfoTooltip>
              <Switch
                checked={showPredictions}
                onCheckedChange={(checked) => {
                  setShowPredictions(checked);
                }}
                disabled={
                  arePredictionsLoading ||
                  isLoading ||
                  dataInput !== "stock_close"
                }
                className="data-[state=checked]:bg-[#ea580c]"
                id="showPreds"
              />
              <Label htmlFor="showPreds">Show Predictions</Label>
            </div>
            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger
                className="w-[160px] rounded-lg sm:ml-auto"
                aria-label="Select a value"
              >
                <SelectValue placeholder="Last 3 months" />
              </SelectTrigger>
              <SelectContent className="rounded-xl">
                <SelectItem value="30d" className="rounded-lg">
                  Last 30 days
                </SelectItem>
                <SelectItem value="14d" className="rounded-lg">
                  Last 14 days
                </SelectItem>
                <SelectItem value="7d" className="rounded-lg">
                  Last 7 days
                </SelectItem>
              </SelectContent>
            </Select>
            <Select
              value={dataInput}
              onValueChange={(value) =>
                setDataInput(value as keyof StockDataItem)
              }
            >
              <SelectTrigger
                className="w-[160px] rounded-lg sm:ml-auto"
                aria-label="Select a data input"
              >
                <SelectValue placeholder="Stock Close" />
              </SelectTrigger>
              <SelectContent className="rounded-xl">
                {[
                  "stock_close",
                  "stock_open",
                  "stock_high",
                  "stock_low",
                  "stock_volume",
                  "sentiment_data",
                  "news_data",
                ].map((item) => (
                  <SelectItem
                    key={"select-" + item}
                    value={item}
                    className="rounded-lg"
                  >
                    {normalizeName(item)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </CardHeader>
          <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
            <ChartContainer
              config={chartConfig}
              className="aspect-auto h-[275px] w-full"
            >
              <AreaChart data={chartData} dataKey="date" accessibilityLayer>
                <defs>
                  {chartConfig &&
                    Object.entries(chartConfig)?.map(([key, val]) => {
                      return (
                        <linearGradient
                          id={key}
                          key={"linear-" + key}
                          x1="0"
                          y1="1"
                          x2="0"
                          y2="1"
                        >
                          <stop
                            offset="5%"
                            stopColor={val.color}
                            stopOpacity={0.8}
                          />
                          <stop
                            offset="95%"
                            stopColor={val.color}
                            stopOpacity={0.15}
                          />
                        </linearGradient>
                      );
                    })}
                </defs>
                {chartData &&
                  Object.keys(chartData[0] ?? {}).map((key, index) => {
                    if (key === "date") return null;
                    return (
                      <Area
                        key={`area-${index}`}
                        type="monotone"
                        dataKey={key}
                        strokeWidth={2}
                        stroke={chartConfig[key].color}
                        fill={`url(#${key})`}
                        activeDot={{ r: 3 }}
                        dot={false}
                        stackId={key}
                      />
                    );
                  })}
                <CartesianGrid vertical={false} />
                <XAxis
                  dataKey="date"
                  tickLine={true}
                  axisLine={true}
                  tickMargin={9}
                  tickFormatter={(value) => moment(value).format("MMM D")}
                />
                <YAxis
                  tickLine={true}
                  axisLine={true}
                  dataKey={dataInput}
                  allowDataOverflow
                  tickMargin={4}
                  tickCount={9}
                  tickFormatter={(value) => {
                    return String(formatNumber(value));
                  }}
                  domain={["auto", "auto"]}
                >
                  <ChartLabel
                    value={YAxisLabels[dataInput]}
                    angle={-90}
                    position="insideLeft"
                    style={{ textAnchor: "middle" }}
                  />
                </YAxis>
                <ChartTooltip
                  cursor={false}
                  content={
                    <ChartTooltipContent
                    // labelFormatter={(val) => moment(val).format("MMM D")}
                    />
                  }
                />

                <ChartLegend content={<ChartLegendContent />} />
              </AreaChart>
            </ChartContainer>
          </CardContent>
        </Card>
      )}
    </>
  );
}
