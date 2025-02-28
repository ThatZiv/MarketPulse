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
// const data = [
//   {
//     "stock_close": 10.15999985,
//     "stock_volume": 56241100,
//     "stock_open": 10.05000019,
//     "stock_high": 10.17000008,
//     "stock_low": 9.970000267,
//     "date": "2025-01-23"
//   },
//   {
//     "stock_close": 10.11999989,
//     "stock_volume": 42909100,
//     "stock_open": 10.15999985,
//     "stock_high": 10.31000042,
//     "stock_low": 10.09000015,
//     "date": "2025-01-24"
//   },
//   {
//     "stock_close": 10.38000011,
//     "stock_volume": 73082600,
//     "stock_open": 10.10999966,
//     "stock_high": 10.42000008,
//     "stock_low": 10.10000038,
//     "date": "2025-01-27"
//   },
//   {
//     "stock_close": 10.07999992,
//     "stock_volume": 92177600,
//     "stock_open": 10.22999954,
//     "stock_high": 10.22999954,
//     "stock_low": 9.920000076,
//     "date": "2025-01-28"
//   },
//   {
//     "stock_close": 10.21000004,
//     "stock_volume": 69921200,
//     "stock_open": 10.10000038,
//     "stock_high": 10.28999996,
//     "stock_low": 10.02999973,
//     "date": "2025-01-29"
//   },
//   {
//     "stock_close": 10.15999985,
//     "stock_volume": 69663100,
//     "stock_open": 10.26000023,
//     "stock_high": 10.39000034,
//     "stock_low": 10.06000042,
//     "date": "2025-01-30"
//   },
//   {
//     "stock_close": 10.07999992,
//     "stock_volume": 92762833,
//     "stock_open": 10.17000008,
//     "stock_high": 10.35000038,
//     "stock_low": 9.989999771,
//     "date": "2025-01-31"
//   },
//   {
//     "stock_close": 9.975199699,
//     "stock_volume": 41674923,
//     "stock_open": 9.720000267,
//     "stock_high": 10,
//     "stock_low": 9.600000381,
//     "date": "2025-02-03"
//   },
//   {
//     "stock_close": 10.15999985,
//     "stock_volume": 64427595,
//     "stock_open": 9.93999958,
//     "stock_high": 10.15999985,
//     "stock_low": 9.904999733,
//     "date": "2025-02-04"
//   },
//   {
//     "stock_close": 10.15499973,
//     "stock_volume": 8888008,
//     "stock_open": 10.15999985,
//     "stock_high": 10.31000042,
//     "stock_low": 10.10999966,
//     "date": "2025-02-05"
//   },
//   {
//     "stock_close": 9.260000229,
//     "stock_volume": 222758300,
//     "stock_open": 9.489999771,
//     "stock_high": 9.640000343,
//     "stock_low": 9.260000229,
//     "date": "2025-02-06"
//   },
//   {
//     "stock_close": 9.239999771,
//     "stock_volume": 140446500,
//     "stock_open": 9.329999924,
//     "stock_high": 9.350000381,
//     "stock_low": 9.180000305,
//     "date": "2025-02-07"
//   },
//   {
//     "stock_close": 9.239999771,
//     "stock_volume": 72168000,
//     "stock_open": 9.25,
//     "stock_high": 9.300000191,
//     "stock_low": 9.119999886,
//     "date": "2025-02-10"
//   },
//   {
//     "stock_close": 9.210000038,
//     "stock_volume": 54844600,
//     "stock_open": 9.210000038,
//     "stock_high": 9.260000229,
//     "stock_low": 9.170000076,
//     "date": "2025-02-11"
//   },
//   {
//     "stock_close": 9.229999542,
//     "stock_volume": 62012399,
//     "stock_open": 9.170000076,
//     "stock_high": 9.289999962,
//     "stock_low": 9.100000381,
//     "date": "2025-02-12"
//   },
//   {
//     "stock_close": 9.350000381,
//     "stock_volume": 71950800,
//     "stock_open": 9.289999962,
//     "stock_high": 9.409999847,
//     "stock_low": 9.239999771,
//     "date": "2025-02-13"
//   },
//   {
//     "stock_close": 9.479999542,
//     "stock_volume": 53545100,
//     "stock_open": 9.430000305,
//     "stock_high": 9.510000229,
//     "stock_low": 9.390000343,
//     "date": "2025-02-14"
//   },
//   {
//     "stock_close": 9.289999962,
//     "stock_volume": 62175800,
//     "stock_open": 9.319999695,
//     "stock_high": 9.350000381,
//     "stock_low": 9.229999542,
//     "date": "2025-02-18"
//   },
//   {
//     "stock_close": 9.340000153,
//     "stock_volume": 50191700,
//     "stock_open": 9.270000458,
//     "stock_high": 9.390000343,
//     "stock_low": 9.210000038,
//     "date": "2025-02-19"
//   },
//   {
//     "stock_close": 9.390000343,
//     "stock_volume": 39787800,
//     "stock_open": 9.329999924,
//     "stock_high": 9.399999619,
//     "stock_low": 9.279999733,
//     "date": "2025-02-20"
//   },
//   {
//     "stock_close": 9.279999733,
//     "stock_volume": 58437700,
//     "stock_open": 9.380000114,
//     "stock_high": 9.409999847,
//     "stock_low": 9.229999542,
//     "date": "2025-02-21"
//   },
//   {
//     "stock_close": 9.350000381,
//     "stock_volume": 72562600,
//     "stock_open": 9.31000042,
//     "stock_high": 9.399999619,
//     "stock_low": 9.210000038,
//     "date": "2025-02-24"
//   },
//   {
//     "stock_close": 9.420000076,
//     "stock_volume": 88876800,
//     "stock_open": 9.369999886,
//     "stock_high": 9.489999771,
//     "stock_low": 9.31000042,
//     "date": "2025-02-25"
//   },
//   {
//     "stock_close": 9.470000267,
//     "stock_volume": 85112536,
//     "stock_open": 9.460000038,
//     "stock_high": 9.619999886,
//     "stock_low": 9.444999695,
//     "date": "2025-02-26"
//   }
// ]

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
