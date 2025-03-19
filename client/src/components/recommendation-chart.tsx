// import { Pie, PieChart, Sector } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer,
  //   ChartTooltip,
  //   ChartTooltipContent,
} from "@/components/ui/chart";
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useGlobal } from "@/lib/GlobalProvider";
import { useApi } from "@/lib/ApiProvider";
import { actions, cache_keys } from "@/lib/constants";
// import { type PieSectorDataItem } from "recharts/types/polar/Pie";
import moment from "moment";
import { Separator } from "./ui/separator";
import { PurchaseHistoryCalculator } from "@/lib/Calculator";

// const chartData = [
//   { action: "buy", suggest: 50, fill: "var(--color-buy)" },
//   { action: "sell", suggest: 20, fill: "var(--color-sell)" },
//   { action: "hold", suggest: 30, fill: "var(--color-hold)" },
// ];

const chartConfig = {
  buy: {
    label: "Buy",
    color: "hsl(var(--chart-1))",
  },
  sell: {
    label: "Sell",
    color: "hsl(var(--chart-2))",
  },
  hold: {
    label: "Hold",
    color: "hsl(var(--chart-3))",
  },
};

interface RecommendationProps {
  stock_ticker: string;
}

export default function Recommendation({ stock_ticker }: RecommendationProps) {
  const { state, dispatch } = useGlobal();
  const api = useApi();
  //   const [, setActiveIndex] = useState<number | null>(null);
  const { data, isError, isLoading } = useQuery({
    queryKey: [cache_keys.STOCK_DATA_REALTIME, stock_ticker],
    refetchInterval: () => 1000 * 60 * 5,
    queryFn: async () => {
      const data = await api?.getStockRealtime(stock_ticker);
      if (!data) return [];
      dispatch({
        type: actions.SET_STOCK_PRICE,
        payload: {
          stock_ticker,
          data: data[data.length - 1].stock_close,
          timestamp: new Date(
            data[data.length - 1].time_stamp.join(" ") + " UTC"
          ).getTime(),
        },
      });
      return data;
    },
    enabled: !!stock_ticker,
  });
  const profit = useMemo(() => {
    if (!data) return undefined;
    const stock = state.stocks[stock_ticker];
    const timeWindow = state.views.predictions.timeWindow;
    if (!stock) return undefined;
    const model = state.views.predictions.model;
    if (!state.predictions[stock_ticker]) return undefined;
    const lastPrice = stock.current_price;
    if (!lastPrice) return undefined;
    const predictions = state.predictions[stock_ticker];
    if (!predictions) return undefined;
    const lookAhead =
      predictions.length - timeWindow >= 0
        ? predictions.length - timeWindow
        : 0;
    const lastPrediction = predictions[lookAhead];
    if (!lastPrediction) return undefined;
    const futurePrices = Object.keys(lastPrediction)
      .filter((key) => key !== "day")
      // if model view is present, we'll only show that model
      .filter((key) => (model ? key == model : key))
      .map((key) => Number(lastPrediction[key]));
    if (!futurePrices.length) return undefined;
    const averageFuturePrice =
      futurePrices.reduce((a, b) => a + b, 0) / futurePrices.length;

    return (averageFuturePrice - lastPrice).toFixed(2);
    // ).map((key) => ({[key]: lastPrediction[key]})); // TODO: weighted average here for models
  }, [
    data,
    state.predictions,
    state.views.predictions,
    state.stocks,
    stock_ticker,
  ]);

  const isLoss = Number(profit) < 0;
  //   const renderActiveShape = (props: PieSectorDataItem) => {
  //     const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill } =
  //       props;

  //     return (
  //       <Sector
  //         cx={cx}
  //         cy={cy}
  //         innerRadius={innerRadius}
  //         outerRadius={outerRadius ?? 0 + 10}
  //         startAngle={startAngle}
  //         endAngle={endAngle}
  //         fill={fill}
  //       />
  //     );
  //   };

  if (isError || (!profit && !isLoading)) {
    return (
      <div>
        No recommendation can be made at this time. Please try again later.
      </div>
    );
  }
  return (
    <Card className="flex flex-col w-full h-full">
      <CardHeader className="items-center pb-0">
        <CardTitle className="font-semibold text-xl">
          Our Recommendation <span className="text-red-500">*</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 pb-0">
        <ChartContainer
          config={chartConfig}
          className="mx-auto aspect-square max-h-[250px]"
        >
          {
            isLoading ? (
              <div className="flex justify-center mt-5">
                <div className="animate-pulse flex space-x-4">
                  <div className="rounded-full bg-gray-200 dark:bg-gray-700 h-32 w-32"></div>
                  <div className="gap-2 flex flex-col justify-center">
                    <div className="rounded-md bg-gray-200 dark:bg-gray-700 w-32 h-2"></div>
                    <div className="rounded-md bg-gray-200 dark:bg-gray-700 w-12 h-2"></div>
                    <div className="rounded-md bg-gray-200 dark:bg-gray-700 w-24 h-2"></div>
                    <div className="rounded-md bg-gray-200 dark:bg-gray-700 w-32 h-2"></div>
                  </div>
                </div>
              </div>
            ) : profit ? (
              // TODO move this to a chart graphical representation
              <div className="min-w-full text-left min-h-full">
                {state.stocks[stock_ticker] && (
                  <div className="text-xs relative justify-center text-center w-full mb-4">
                    As of{" "}
                    {moment(state.stocks[stock_ticker].timestamp).calendar()}
                  </div>
                )}
                <Separator className="my-2" />
                <span className="text-sm text-left">
                  Price change in {state.views.predictions.timeWindow} day(s) is
                  expected to be:{" "}
                </span>
                <div
                  className={`md:text-4xl mr-4 text-center ${
                    isLoss ? "text-red-700" : "text-green-700"
                  }`}
                >
                  {PurchaseHistoryCalculator.toDollar(Number(profit))}
                </div>
                <div className="justify-center w-full mb-4">
                  <div className="text-sm text-left">
                    It might be a good idea to{" "}
                  </div>
                  <div className="text-2xl font-bold text-left">
                    {isLoss ? "Sell" : "Buy"} {stock_ticker}.
                  </div>
                </div>
              </div>
            ) : (
              <div>
                No recommendation can be made at this time. Please try again
                later.
              </div>
            )

            // <PieChart>
            //   <ChartTooltip
            //     cursor={false}
            //     content={<ChartTooltipContent hideLabel />}
            //   />
            //   <Pie
            //     data={chartData}
            //     dataKey="suggest"
            //     nameKey="action"
            //     innerRadius={60}
            //     outerRadius={80}
            //     activeShape={renderActiveShape}
            //     onMouseEnter={(_, index) => setActiveIndex(index)}
            //     onMouseLeave={() => setActiveIndex(null)}
            //     className="stroke-transparent stroke-2 hover:stroke-[0.3rem]"
            //     style={{ transition: "stroke 1s" }}
            //   />
            // </PieChart>
          }
        </ChartContainer>
      </CardContent>
      {/* <div className="chart-labels flex justify-around mt-4 pb-3">
        <div className="label-item flex items-center space-x-2">
          <div
            className="w-4 h-4"
            style={{ backgroundColor: chartConfig.buy.color }}
          ></div>
          <span style={{ color: chartConfig.buy.color }}>
            {chartConfig.buy.label}
          </span>
        </div>
        <div className="label-item flex items-center space-x-2">
          <div
            className="w-4 h-4"
            style={{ backgroundColor: chartConfig.sell.color }}
          ></div>
          <span style={{ color: chartConfig.sell.color }}>
            {chartConfig.sell.label}
          </span>
        </div>
        <div className="label-item flex items-center space-x-2">
          <div
            className="w-4 h-4"
            style={{ backgroundColor: chartConfig.hold.color }}
          ></div>
          <span style={{ color: chartConfig.hold.color }}>
            {chartConfig.hold.label}
          </span>
        </div>
      </div> */}
    </Card>
  );
}
