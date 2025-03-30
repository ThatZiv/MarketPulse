"use client";

import * as React from "react";
import { Area, AreaChart, CartesianGrid, Label, XAxis, YAxis } from "recharts";

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
import { capitalizeFirstLetter } from "@/lib/utils";
interface StockChartProps {
  ticker: string;
}

// FIXME:
// - area gradient not working
// - ChartToolTipContent invalid date
// - chart data not stacking
// - Y label cut off

// TODO:
// - add prediction data

export default function HistoricalChart({ ticker }: StockChartProps) {
  const [timeRange, setTimeRange] = React.useState("7d");
  const api = useApi();
  const { dispatch } = useGlobal();
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
        date: item.time_stamp,
        ...item,
      }));
    },
    enabled: !!api && !!ticker,
  });

  const timeRangeValue = React.useMemo(() => {
    return parseInt(timeRange.match(/(\d+)/)?.[0] || "0");
  }, [timeRange]);

  const chartData = React.useMemo(() => {
    if (!data) return [];
    const filteredData = data?.map((item) => ({
      date: new Date(item.time_stamp.join(" ")),
      "stock open": item.stock_open,
      "stock close": item.stock_close,
    }));
    return filteredData.slice(
      filteredData.length - timeRangeValue,
      filteredData.length
    );
  }, [data, timeRangeValue]);

  const chartConfig = React.useMemo<ChartConfig>(() => {
    const config: ChartConfig = {};
    const colors = ["#E2C541", "#92E98C", "#479BC6", "#ea580c"];
    if (!chartData) return config;
    for (const name of Object.keys(chartData[0] ?? {}) ?? []) {
      if (name === "date") continue;
      config[name] = {
        // label: name.split("_").map(capitalizeFirstLetter).join(" "),
        label: name,
        color: colors.shift() ?? "#606060",
      };
    }
    return config;
  }, [chartData]);

  return (
    <>
      {isLoading ? (
        <Card className="w-full p-4 border border-black dark:border-white ">
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
              <AreaChart data={chartData}>
                <defs>
                  {chartConfig &&
                    Object.values(chartConfig)?.map((item) => {
                      return (
                        <linearGradient
                          id={item.label as string}
                          x1="0"
                          y1="0"
                          x2="0"
                          y2="1"
                        >
                          <stop
                            offset="5%"
                            stopColor={item.color}
                            stopOpacity={0.8}
                          />
                          <stop
                            offset="95%"
                            stopColor={item.color}
                            stopOpacity={0.1}
                          />
                        </linearGradient>
                      );
                    })}
                </defs>
                <CartesianGrid vertical={false} />
                <YAxis
                  tickLine={true}
                  axisLine={true}
                  tickMargin={5}
                  domain={["auto", "auto"]}
                >
                  <Label
                    value="Stock Price ($)"
                    angle={-90}
                    position="insideLeft"
                    offset={-5}
                    style={{ textAnchor: "middle" }}
                  />
                </YAxis>
                <XAxis
                  dataKey="date"
                  tickLine={true}
                  allowDuplicatedCategory={false}
                  axisLine={true}
                  tickMargin={9}
                  tickFormatter={(value) => moment(value).format("MMM D")}
                />
                <ChartTooltip
                  cursor={false}
                  content={
                    <ChartTooltipContent
                      labelFormatter={(val) => moment(val).format("MMM D")}
                    />
                  }
                />

                {chartData &&
                  Object.keys(chartData[0] ?? {}).map((key, index) => {
                    if (key === "date") return null;
                    return (
                      <Area
                        key={index}
                        type="monotone"
                        dataKey={key}
                        strokeWidth={2}
                        stroke={chartConfig[key].color}
                        fill={`url(#${chartConfig[key].label})`}
                        activeDot={{ r: 3 }}
                        dot={false}
                        stackId="a"
                      />
                    );
                  })}
                <ChartLegend content={<ChartLegendContent />} />
              </AreaChart>
            </ChartContainer>
          </CardContent>
        </Card>
      )}
    </>
  );
}
