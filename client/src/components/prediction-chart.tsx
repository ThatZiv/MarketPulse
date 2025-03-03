import React from "react";
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
import { Skeleton } from "@/components/ui/skeleton";

// Chart configuration
const chartConfig = {
  predicted_price: {
    label: "Predicted Price",
    color: "hsl(var(--chart-1))",
  },
} satisfies ChartConfig;

type Predictions = {
  date: string;
  predicted_price: number;
};

type Props = {
  ticker: string;
  predictions: Predictions[];
};

export default function Stock_Chart({ ticker, predictions }: Props) {
  const [timeRange, setTimeRange] = React.useState("14d");

  const validData = predictions ?? ([] as Predictions[]);

  const labels = validData.map((item) => item.date);
  const predicted_price = validData.map((item) => item.predicted_price);

  const chartData = labels.map((label, index) => ({
    date: label,
    predicted_price: predicted_price[index],
  }));

  // Filter the data based on the selected time range
  const filteredData = React.useMemo(() => {
    return chartData.filter((item, index) => {
      if (timeRange === "14d") {
        return chartData.length - 14 <= index;
      }
      if (timeRange === "7d") {
        return chartData.length - 7 <= index;
      }
      return true;
    });
  }, [timeRange, chartData]);

  return (
    <Card className="w-full border border-black dark:border-white ">
      <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
        <div className="grid flex-1 gap-1 text-center sm:text-left">
          <CardTitle>{ticker}'s Predictions</CardTitle>
          <CardDescription>
            <div className="text-xs">
              Updated as of{" "}
              {new Date(chartData[chartData.length - 1]?.date).toLocaleDateString()}
            </div>
          </CardDescription>
        </div>
        <Select value={timeRange} onValueChange={setTimeRange}>
          <SelectTrigger className="w-[160px] rounded-lg sm:ml-auto" aria-label="Select a value">
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
        <ChartContainer config={chartConfig} className="aspect-auto h-[250px] w-full">
          <AreaChart data={filteredData}>
            <defs>
              <linearGradient id="fillpredicted_price" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--color-predicted_price)" stopOpacity={0.8} />
                <stop offset="95%" stopColor="var(--color-predicted_price)" stopOpacity={0.1} />
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
            <YAxis tickLine={true} axisLine={true} tickMargin={9} tickCount={9} domain={["auto", "auto"]}>
              <Label value="Stock Price" angle={-90} position="insideLeft" style={{ textAnchor: "middle" }} />
            </YAxis>

            <ChartTooltip
              cursor={true}
              content={<ChartTooltipContent labelFormatter={(value) => new Date(value).toLocaleDateString("en-US", { month: "short", day: "numeric" })} />}
            />
            <Area
              dataKey="predicted_price"
              type="natural"
              fill="url(#fillpredicted_price)"
              stroke="var(--color-predicted_price)"
              stackId="a"
            />
            <ChartLegend content={<ChartLegendContent />} />
          </AreaChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
