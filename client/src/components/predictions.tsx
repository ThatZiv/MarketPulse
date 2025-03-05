import { cache_keys } from "@/lib/constants";
import React from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { CartesianGrid, Label, Line, LineChart, XAxis, YAxis } from "recharts";
import moment from "moment";

import {
  ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { useApi } from "@/lib/ApiProvider";
// const chartData = [
//   { day: "January", desktop: 186, mobile: 80 },
//   { day: "February", desktop: 305, mobile: 200 },
//   { day: "March", desktop: 237, mobile: 120 },
//   { day: "April", desktop: 73, mobile: 190 },
//   { day: "May", desktop: 209, mobile: 130 },
//   { day: "June", desktop: 214, mobile: 140 },
// ];

interface PredictionsProps {
  stock_id: number;
  stock_name: string;
  stock_ticker: string;
}

// type StockPrediction = {
//   model_name: string;
//   prediction: number[];
//   timestamp: number;
// };

export default function Predictions({
  stock_id,
  stock_name,
  stock_ticker,
}: PredictionsProps) {
  // const { supabase } = useSupabase();
  const api = useApi();
  const { data, isLoading, isError } = useQuery({
    queryKey: [cache_keys.STOCK_PREDICTION, stock_id],
    queryFn: async () => {
      // const data = await supabase
      //   .from("Stock_Predictions")
      //   .select("*")
      //   .eq("stock_id", stock_id);
      const resp = await api?.getStockPredictions(stock_ticker);

      if (!resp) {
        throw new Error("Failed to fetch stock predictions");
      }
      return resp;
    },
    enabled: !!stock_id && !!api,
  });
  const predictions = data?.output;

  const chartData = React.useMemo(() => {
    const points: Array<Datapoint> = [];
    if (!predictions) return points;

    // TODO: make sure this works when:
    // - predictions are made on a weekend
    // - predictions are made on a friday
    //// predictions are made on a monday
    let startDate = moment(new Date(data.created_at));

    // initial weekend day check
    if (startDate.isoWeekday() === 6) {
      // saturday -> monday
      startDate = startDate.add(2, "days");
    } else if (startDate.isoWeekday() === 7) {
      // sunday -> monday
      startDate = startDate.add(1, "days");
    }

    const currentDate = startDate.clone();

    for (const i in predictions[0].forecast) {
      // skip weekends on run
      while (currentDate.isoWeekday() > 5) {
        currentDate.add(1, "days");
      }

      const point: Datapoint = {
        day: currentDate.format("MMM DD"),
      };

      for (const { name, forecast } of predictions) {
        point[name] = forecast[i];
      }

      points.push(point);

      currentDate.add(1, "days");
      // if (currentDate.isoWeekday() === 6) {
      //   currentDate.add(2, "days");
      // } else if (currentDate.isoWeekday() === 7) {
      //   currentDate.add(1, "days");
      // }
    }

    return points;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predictions]);

  const chartConfig = React.useMemo(() => {
    const config: ChartConfig = {};
    const colors = ["#F5A79F", "#E2C541", "#BF0F52", "#92E98C", "#479BC6"];
    for (const { name } of predictions ?? []) {
      config[name] = {
        label: name,
        color: colors.shift() ?? "#606060",
      };
    }
    return config;
  }, [predictions]);

  type Datapoint = Record<string, string | number>;
  if (!stock_id) {
    return <div>Stock ID not found</div>;
  }

  if (!predictions && !isLoading) {
    return (
      <div>
        No predictions currently available for {stock_name}. Please wait until
        tomorrow.
      </div>
    );
  }

  if (isError) {
    return <div>Failed to fetch stock predictions</div>;
  }

  return (
    <Card className="w-full border border-black dark:border-white ">
      {isLoading ? (
        <div
          role="status"
          className="border border-gray-200 rounded-sm shadow-sm animate-pulse p-2 dark:border-gray-700"
        >
          <div className="w-12 h-2 my-2 bg-gray-200 rounded-full dark:bg-gray-700"></div>
          <div className="w-full h-2 my-2 bg-gray-200 rounded-full dark:bg-gray-700"></div>
          <div className="w-full h-2 my-2 bg-gray-200 rounded-full dark:bg-gray-700"></div>
          <div className="flex items-center justify-center">
            <div className="h-48 w-full bg-gray-200 dark:bg-gray-700"></div>
          </div>
        </div>
      ) : (
        <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
          <div className="grid flex-1 gap-1 text-center sm:text-left">
            <CardTitle>{stock_ticker} Forecasts</CardTitle>
            <CardDescription>
              <div className="text-xs ">
                Up to {chartData[chartData.length - 1]?.day}
              </div>
            </CardDescription>

            <CardContent className="flex items-center justify-center">
              <ChartContainer
                config={chartConfig}
                className="min-h-[300px] w-full"
              >
                <LineChart
                  accessibilityLayer
                  data={chartData}
                  margin={{
                    left: 12,
                    right: 12,
                  }}
                >
                  <CartesianGrid vertical={false} />
                  <XAxis
                    dataKey="day"
                    tickLine={true}
                    allowDuplicatedCategory={false}
                    axisLine={true}
                    tickMargin={9}
                  />
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
                    cursor={true}
                    content={<ChartTooltipContent />}
                  />
                  <ChartLegend content={<ChartLegendContent />} />
                  {Object.keys(chartData[chartData.length - 1] ?? []).map(
                    (key, index) => {
                      if (key === "day") return null;
                      return (
                        <Line
                          key={index}
                          type="monotone"
                          dataKey={key}
                          strokeWidth={2}
                          stroke={chartConfig[key].color}
                          dot={true}
                          activeDot={{ r: 3 }}
                        />
                      );
                    }
                  )}
                </LineChart>
              </ChartContainer>
            </CardContent>
          </div>
        </CardHeader>
      )}
    </Card>
  );
}
