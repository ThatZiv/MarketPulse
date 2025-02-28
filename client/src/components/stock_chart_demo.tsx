"use client"

import React, { useState, useEffect } from "react"
import { Area, AreaChart, CartesianGrid, XAxis, YAxis, Label  } from "recharts"
import axios from "axios"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

type stock_Data = {
  sentiment_data: number;
  stock_close: number;
  stock_high: number;
  stock_id: number;
  stock_low: number;
  stock_open: number;
  stock_volume: number;
  time_stamp: string[];
}[];

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
} satisfies ChartConfig
type props = { ticker: string };
export default function Stock_Chart(props: props) {
  const [stockData, setStockData] = useState<stock_Data>([]);

  const [isDataFetched, setIsDataFetched] = useState(false);
  const [timeRange, setTimeRange] = React.useState("14d");

  useEffect(() => {
    const authToken = localStorage.getItem("sb-xskipmrkpwewdbttinhd-auth-token");

    if (authToken && !isDataFetched) {
      const token = JSON.parse(authToken);

      const chartData = async () => {
        try {
          const response = await axios.get(
            `http://127.0.0.1:5000/stockchart/?ticker=${props.ticker}`,
            {
              method: "get",
              headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${token.access_token}`,
              },
            }
          );

          console.log("API Response:", response);
          setStockData(response.data);
          setIsDataFetched(true);
        } catch (error) {
          console.error("Error:", error);
        }
      };

      chartData();
    } else {
      console.log("Failed to find auth token or data already fetched.");
    }
  }, [props.ticker, isDataFetched]);

  useEffect(() => {
    if (stockData.length > 0) {
      console.log("Stock Data:", stockData);
      console.log("Data Length:", stockData.length);
      console.log("Time Stamp:", stockData[stockData.length - 15].time_stamp[0] || "");
    }
    else{
      console.log("No stock data available.");
    }
  }, [stockData]);

  const labels = stockData.length >= 15
    ? stockData.slice(-15).map(item => item.time_stamp[0])
    : [];
  const close_values = stockData.length >= 15
    ? stockData.slice(-15).map(item => item.stock_close)
    : [];
    const open_values = stockData.length >= 15
    ? stockData.slice(-15).map(item => item.stock_open)
    : [];
    const high_values = stockData.length >= 15
    ? stockData.slice(-15).map(item => item.stock_high)
    : [];
    const low_values = stockData.length >= 15
    ? stockData.slice(-15).map(item => item.stock_low)
    : [];
  const data = labels.map((label, index) => ({
    date: label,
    close: close_values[index],
    open: open_values[index],
    high: high_values[index],
    low: low_values[index],
  }))
  console.log("data: ", data);

  const filteredData = data.filter((item) => {
    if (timeRange === "14d") {
      return data.length - 14 <= data.indexOf(item);
    }
    if (timeRange === "7d") {
      return data.length - 7 <= data.indexOf(item);
    }
  });
  return (
    <Card className="w-full">
      <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
        <div className="grid flex-1 gap-1 text-center sm:text-left">
          <CardTitle>Historical Stock Data for {props.ticker}</CardTitle>
          <CardDescription>
            {data[0]?.date} - {data[data.length - 1]?.date}
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
              domain ={['auto','auto']}
            >
              <Label
                value="Stock Price"
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
  );
}
