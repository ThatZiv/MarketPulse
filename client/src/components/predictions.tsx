import { useSupabase } from "@/database/SupabaseProvider";
import { cache_keys } from "@/lib/constants";
import { useQuery } from "@tanstack/react-query";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { CartesianGrid, Line, LineChart, XAxis } from "recharts";
import moment from "moment";

import {
  ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Spinner } from "./ui/spinner";
import { useApi } from "@/lib/ApiProvider";
// const chartData = [
//   { month: "January", desktop: 186, mobile: 80 },
//   { month: "February", desktop: 305, mobile: 200 },
//   { month: "March", desktop: 237, mobile: 120 },
//   { month: "April", desktop: 73, mobile: 190 },
//   { month: "May", desktop: 209, mobile: 130 },
//   { month: "June", desktop: 214, mobile: 140 },
// ];

// TODO: make this dynamic
const chartConfig = {
  attention_lstm: {
    label: "attention_lstm",
    color: "#FF0000",
  },
  "cnn-lstm-transformer": {
    label: "cnn-lstm-transformer",
    color: "#00FF00",
  },
  "zav-transformer": {
    label: "zav-transformer",
    color: "#0000FF",
  },
  model_4: {
    label: "model_4",
    color: "#FFFF00",
  },
  model_5: {
    label: "model_5",
    color: "#FF00FF",
  },
} satisfies ChartConfig;

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
  const { supabase } = useSupabase();
  const api = useApi();
  const { data, isLoading, isError } = useQuery({
    queryKey: [cache_keys.STOCK_PREDICTION, stock_id],
    queryFn: async () => {
      // const data = await supabase
      //   .from("Stock_Predictions")
      //   .select("*")
      //   .eq("stock_id", stock_id);
      const resp = await api?.getStockPredictions(stock_ticker);
      console.log(resp);
      if (!resp) {
        throw new Error("Failed to fetch stock predictions");
      }
      return resp;
    },
    enabled: !!stock_id,
  });
  if (!stock_id) {
    return <div>Stock ID not found</div>;
  }
  if (!data || isLoading) {
    return <Spinner />;
  }
  const predictions = data?.data;
  if (!predictions) {
    return <div>No predictions currently available for {stock_name}</div>;
  }
  if (isError) {
    return <div>Failed to fetch stock predictions</div>;
  }
  const points = [];
  const chartData = predictions.map(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (prediction: Record<string, any>) => {
      const point = {
        month: new Date(prediction.created_at).toDateString(),
        // month: index,
      };
      for (const [key, value] of Object.entries(
        prediction as Record<string, unknown>
      )) {
        if (key.startsWith("model_")) {
          const jsonb_content = JSON.parse(value as string);
          for (const price of jsonb_content.forecast) {
            point["month"] = moment(jsonb_content.month)
              .add(1, "days")
              .format("YYYY-MM-DD");
            // @ts-expect-error any

            point[jsonb_content.name as string] = price;
          }
        }
        points.push(point);
      }
      return point;
    }
  );
  //   console.log(points);
  return (
    <Card className="w-full border border-black dark:border-white ">
      {isLoading ? (
        <Spinner />
      ) : (
        <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
          <div className="grid flex-1 gap-1 text-center sm:text-left">
            <CardTitle>{stock_ticker} Forecasts</CardTitle>
            <CardDescription>
              <div className="text-xs ">
                Up to {chartData[chartData.length - 1]?.month}
              </div>
            </CardDescription>

            <CardContent className="flex items-center justify-center">
              <ChartContainer
                config={chartConfig}
                className="min-h-[200px] w-full"
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
                    dataKey="month"
                    tickLine={true}
                    axisLine={true}
                    tickMargin={8}
                  />
                  <ChartTooltip
                    cursor={false}
                    content={<ChartTooltipContent />}
                  />
                  <ChartLegend content={<ChartLegendContent />} />
                  {Object.keys(chartData[0]).map((key, index) => {
                    if (key === "month") return null;
                    return (
                      <Line
                        key={index}
                        type="monotone"
                        dataKey={key}
                        strokeWidth={2}
                        dot={true}
                        activeDot={{ r: 3 }}
                      />
                    );
                  })}
                </LineChart>
              </ChartContainer>
            </CardContent>
          </div>
        </CardHeader>
      )}
    </Card>
  );
}
