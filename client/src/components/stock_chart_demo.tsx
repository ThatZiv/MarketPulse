import React from "react";
import { useQuery } from "@tanstack/react-query";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis, Label } from "recharts";
import { StockDataItem } from "@/types/stocks";
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
import { actions, cache_keys } from "@/lib/constants";
import { useGlobal } from "@/lib/GlobalProvider";

const chartConfig = {
  close: {
    label: "Close",
    color: "hsl(var(--chart-1))",
  },
  open: {
    label: "Open",
    color: "hsl(var(--chart-2))",
  },
  high: {
    label: "High",
    color: "hsl(var(--chart-3))",
  },
  low: {
    label: "Low",
    color: "hsl(var(--chart-4))",
  },
} satisfies ChartConfig;

type props = { ticker: string };
export default function Stock_Chart({ ticker }: props) {
  const api = useApi();
  const { dispatch } = useGlobal();
  const [timeRange, setTimeRange] = React.useState("14d");
  // const timeRangeNum = React.useMemo(
  //   () => Number(timeRange.match("[0-9]+")?.[0]),
  //   [timeRange]
  // );
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
      return data;
    },
    enabled: !!api && !!ticker,
  });

  const validData = data ?? ([] as StockDataItem[]);

  const labels =
    validData.length >= 15
      ? validData.slice(-15).map((item) => item.time_stamp[0])
      : [];
  const close_values =
    validData.length >= 15
      ? validData.slice(-15).map((item) => item.stock_close)
      : [];
  const open_values =
    validData.length >= 15
      ? validData.slice(-15).map((item) => item.stock_open)
      : [];
  const high_values =
    validData.length >= 15
      ? validData.slice(-15).map((item) => item.stock_high)
      : [];
  const low_values =
    validData.length >= 15
      ? validData.slice(-15).map((item) => item.stock_low)
      : [];
  const chartData = labels.map((label, index) => ({
    date: label,
    close: close_values[index],
    open: open_values[index],
    high: high_values[index],
    low: low_values[index],
  }));

  const filteredData = chartData.filter((item) => {
    if (timeRange === "14d") {
      return chartData.length - 14 <= chartData.indexOf(item);
    }
    if (timeRange === "7d") {
      return chartData.length - 7 <= chartData.indexOf(item);
    }
  });

  return (
    <>
      {isLoading || isError ? (
        <Card className="w-full p-4">
          {isLoading && (
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
          {isError && (
            <div>Error fetching stock data. Please try again later...</div>
          )}
        </Card>
      ) : (
        <Card className="w-full border border-black dark:border-white ">
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
            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger
                className="w-[160px] rounded-lg sm:ml-auto"
                aria-label="Select a value"
              >
                <SelectValue placeholder="Last 3 months" />
              </SelectTrigger>
              <SelectContent className="rounded-xl">
                <SelectItem value="14d" className="rounded-lg">
                  Last 14 days
                </SelectItem>
                <SelectItem value="7d" className="rounded-lg">
                  Last 7 days
                </SelectItem>
              </SelectContent>
            </Select>
          </CardHeader>
          <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
            <ChartContainer
              config={chartConfig}
              className="aspect-auto h-[250px] w-full"
            >
              <AreaChart data={filteredData}>
                <defs>
                  <linearGradient id="fillOpen" x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="5%"
                      stopColor="var(--color-open)"
                      stopOpacity={0.8}
                    />
                    <stop
                      offset="95%"
                      stopColor="var(--color-open)"
                      stopOpacity={0.1}
                    />
                  </linearGradient>
                  <linearGradient id="fillClose" x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="5%"
                      stopColor="var(--color-close)"
                      stopOpacity={0.8}
                    />
                    <stop
                      offset="95%"
                      stopColor="var(--color-close)"
                      stopOpacity={0.1}
                    />
                  </linearGradient>
                  <linearGradient id="fillHigh" x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="5%"
                      stopColor="var(--color-high)"
                      stopOpacity={0.8}
                    />
                    <stop
                      offset="95%"
                      stopColor="var(--color-high)"
                      stopOpacity={0.1}
                    />
                  </linearGradient>
                  <linearGradient id="fillLow" x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="5%"
                      stopColor="var(--color-low)"
                      stopOpacity={0.8}
                    />
                    <stop
                      offset="95%"
                      stopColor="var(--color-low)"
                      stopOpacity={0.1}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid vertical={false} />
                <XAxis
                  dataKey="date"
                  tickLine={true}
                  axisLine={true}
                  tickMargin={8}
                  minTickGap={32}
                  tickFormatter={(value) => {
                    const date = new Date(value);
                    return date.toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric",
                    });
                  }}
                >
                  <Label value="Date" offset={-10} position="insideBottom" />
                </XAxis>
                <YAxis
                  tickLine={true}
                  axisLine={true}
                  tickMargin={9}
                  tickCount={9}
                  domain={["auto", "auto"]}
                >
                  <Label
                    value="Stock Price ($)"
                    angle={-90}
                    position="insideLeft"
                    style={{ textAnchor: "middle" }}
                  />
                </YAxis>

                <ChartTooltip
                  cursor={false}
                  content={
                    <ChartTooltipContent
                      labelFormatter={(value) => {
                        return new Date(value).toLocaleDateString("en-US", {
                          month: "short",
                          day: "numeric",
                        });
                      }}
                      indicator="dot"
                    />
                  }
                />
                {/* Will Add lines for open, high and low later */}
                <Area
                  dataKey="close"
                  type="natural"
                  fill="url(#fillClose)"
                  stroke="var(--color-close)"
                  stackId="a"
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
