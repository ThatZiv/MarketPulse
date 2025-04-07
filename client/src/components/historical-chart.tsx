import * as React from "react";
import {
  Area,
  CartesianGrid,
  Label as ChartLabel,
  ComposedChart,
  Line,
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
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useApi } from "@/lib/ApiProvider";
import { useGlobal } from "@/lib/GlobalProvider";
import { useQuery } from "@tanstack/react-query";
import { StockDataItem } from "@/types/stocks";
import { actions, cache_keys } from "@/lib/constants";
import moment from "moment";
import {
  capitalizeFirstLetter,
  expandDomain,
  formatNumber,
  isSameDay,
} from "@/lib/utils";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import InfoTooltip from "./InfoTooltip";
import { ChartDatapoint } from "@/types/global_state";
import { Button } from "./ui/button";
import HistoricalAccuracy from "./historical-accuracy";
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
  const [isAdvanced, setAdvanced] = React.useState(false);
  const [isCondensed, setCondensed] = React.useState(true);
  const [cursorForecast, setCursorForecast] = React.useState<ChartDatapoint[]>(
    []
  );

  // chart data eventually takes a union with cursorForecast
  const [chartData, setChartData] = React.useState<ChartDatapoint[]>([]);
  const [cursor, setCursor] = React.useState<string | null>(null);
  const { state, dispatch } = useGlobal();

  // data labels for chart 'dataKey'
  const [dataKeyInput, setDataKeyInput] =
    React.useState<keyof StockDataItem>("stock_close");
  const [cursorKeyInput, setCursorKeyInput] = React.useState<string | null>(
    null
  );
  const api = useApi();
  const [showPredictions, setShowPredictions] = React.useState(false);

  const timeRangeValue = React.useMemo(() => {
    return parseInt(timeRange.match(/(\d+)/)?.[0] || "0");
  }, [timeRange]);

  React.useEffect(() => {
    // reset when page switch
    return () => {
      setShowPredictions(false);
      setDataKeyInput("stock_close");
      resetCursor();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

  const resetCursor = React.useCallback(() => {
    setCursorForecast([]);
    setCursorKeyInput(null);
    setCursor(null);
  }, []);

  React.useEffect(() => {
    // side effect to get the historical forecast for the last n days for x model

    if (!data) return;
    if (!predictionHistory) return;
    const filteredData = data?.map((item) => {
      const thisDate = moment(
        new Date(item.time_stamp.join(" ") + " EST")
      ).toDate();

      const thisPredictionHistory = predictionHistory.find((point) => {
        const tday = moment(point.created_at + " EST");
        return isSameDay(
          thisDate,
          // we need to skip fridays to align w/ historical
          tday.add(tday.weekday() == 5 ? 3 : 1, "days").toDate()
        );
      });
      // @ts-expect-error initially wont have data
      const point: ChartDatapoint = {
        date: thisDate,
      };
      if (dataKeyInput !== "stock_close") {
        setShowPredictions(false);
      }
      point[dataKeyInput] = item[dataKeyInput] as number;
      if (showPredictions) {
        const forecast = thisPredictionHistory?.output.find(
          (point) => point.name === (state.views.predictions.model || "average")
        )?.forecast;
        if (!forecast) return point; // this should never happen
        point[`${state.views.predictions.model || "average"}_prediction`] =
          forecast[0];
      }
      return point;
    });
    setChartData(
      filteredData.slice(
        filteredData.length - timeRangeValue,
        filteredData.length
      )
    );
  }, [
    data,
    timeRangeValue,
    predictionHistory,
    showPredictions,
    dataKeyInput,
    state.views.predictions.model,
  ]);
  React.useEffect(() => {
    // side effect to get the forecast for the cursor date

    const doCursorForecast = async () => {
      if (!isAdvanced) return; // only run if advanced view is enabled
      // get current cursor date
      const startDate = cursor && new Date(cursor);
      if (!startDate || !predictionHistory) return;
      // find the forecast for that date
      const thisForecastHistory = predictionHistory?.find((point) => {
        const today = moment(point.created_at + " EST");
        return isSameDay(
          startDate,
          // we need to skip fridays to align w/ historical
          today.add(today.weekday() == 5 ? 3 : 1, "days").toDate()
        );
      });
      if (!thisForecastHistory) return;
      const today = moment(startDate);
      const todayCursorLabel = moment(startDate).format("MMMM_D") + "_forecast";
      setCursorKeyInput(todayCursorLabel);
      const filteredForecast = thisForecastHistory?.output
        .find(
          (point) => point.name === (state.views.predictions.model || "average")
        )
        ?.forecast.map((val, index) => {
          // add only after first forecast date
          if (index > 0) today.add(today.weekday() == 5 ? 3 : 1, "days");
          return {
            date: today.toDate(),
            [todayCursorLabel]: val,
          } as unknown as ChartDatapoint;
        });
      // we do the data merge with historical predictions in chartData
      if (!filteredForecast) return;
      setCursorForecast(filteredForecast);
    };

    doCursorForecast();

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cursor, predictionHistory]);

  React.useEffect(() => {
    // reset cursor when model changes
    resetCursor();
  }, [state.views.predictions.model, resetCursor]);

  const chartConfig = React.useMemo<ChartConfig>(() => {
    const config: ChartConfig = {};
    // TODO: match the model colors from predictions to this below
    // const colors = ["#479BC6", ...model_colors.reverse()];
    const colors = ["#479BC6", "#ea580c", "#f0c929"];
    if (!chartData) return config;

    for (const name of expandDomain(chartData.concat(cursorForecast))) {
      if (name === "date") continue;
      config[name] = {
        label: normalizeName(name),
        color: colors.shift() ?? "#606060",
      };
    }
    return config;
  }, [chartData, cursorForecast]);

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
        <Card
          className="w-full p-4 border border-black dark:border-white"
          // onMouseLeave={() => setCursor(null)}
        >
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
            {isAdvanced && (
              <div className="flex items-center space-x-2">
                <InfoTooltip side="left">
                  <div className="text-sm">
                    Toggle to show/hide predictions on the chart. Initially,
                    average predictions are based on the average output from all
                    the forecast models.{" "}
                    <span className="font-bold">
                      You can view predictions only when viewing stock closing
                    </span>{" "}
                    because predictions are based on stock closing prices.
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
                    dataKeyInput !== "stock_close"
                  }
                  className="data-[state=checked]:bg-[#ea580c]"
                  id="showPreds"
                />
                <Label htmlFor="showPreds">Show Predictions</Label>
              </div>
            )}
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
            {isAdvanced && (
              <Select
                value={dataKeyInput}
                onValueChange={(value) =>
                  setDataKeyInput(value as keyof StockDataItem)
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
            )}
          </CardHeader>
          <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
            <ChartContainer
              config={chartConfig}
              className="aspect-auto h-[275px] w-full"
            >
              <ComposedChart
                onClick={(chartEvent) => {
                  if (!chartEvent || !chartEvent.activeLabel) return;
                  const clickedDate = chartEvent.activeLabel;
                  setCursorForecast([]);
                  setCursorKeyInput(null);
                  setCursor(clickedDate);
                }}
                data={chartData.map((point) => {
                  const pointObj = point as Partial<ChartDatapoint>;
                  const forecastObj =
                    cursorForecast?.find((f) =>
                      isSameDay(f.date, new Date(point.date))
                    ) || {};

                  return {
                    ...pointObj,
                    ...forecastObj,
                  };
                })}
                dataKey="date"
                accessibilityLayer
              >
                <defs>
                  {chartConfig &&
                    Object.entries(chartConfig)?.map(([key, val]) => {
                      return (
                        <linearGradient
                          id={key}
                          key={"linear-" + key}
                          x1="0"
                          y1="0"
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
                  expandDomain(chartData ?? []).map((key, index) => {
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

                {cursor &&
                  cursorForecast &&
                  showPredictions &&
                  cursorKeyInput &&
                  isAdvanced && (
                    <Line
                      type="monotone"
                      dataKey={cursorKeyInput}
                      stroke="#ff0000" // Use a distinct color for the forecast line
                      strokeWidth={2}
                      activeDot={{ r: 5 }}
                    />
                  )}
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
                  dataKey={isCondensed ? dataKeyInput : undefined}
                  allowDataOverflow
                  tickMargin={4}
                  tickCount={9}
                  tickFormatter={(value) => {
                    return String(formatNumber(value));
                  }}
                  domain={["auto", "auto"]}
                >
                  <ChartLabel
                    value={YAxisLabels[dataKeyInput]}
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
              </ComposedChart>
            </ChartContainer>
            <div className="flex items-center justify-end gap-2 space-y-0 py-2 sm:flex-row">
              <div className="flex items-center space-x-2">
                <InfoTooltip side="left">
                  <span className="text-sm">
                    <strong>
                      Toggle to enable advanced view. Advanced view allows you
                      to:
                    </strong>
                    <div className="list-disc list-inside">
                      <li>
                        View the historical predictions, as well as for a
                        specific date. Click on the chart to view the forecast
                        for that date.
                      </li>
                      <li>
                        Select a model to view the forecast. By default, the
                        chart shows the average forecast from all models.
                      </li>
                      <li>
                        View the historical accuracy of the model. This is only
                        available when viewing stock closing prices.
                      </li>
                      <li>
                        Select a different data point to view on the chart. By
                        default, the chart shows the stock closing price, but
                        you can view others like high/low price, volume,
                        sentiment data, and more.
                      </li>
                    </div>
                  </span>
                </InfoTooltip>
                <Switch
                  checked={isAdvanced}
                  onCheckedChange={(checked) => {
                    resetCursor();
                    setAdvanced(checked);
                    if (checked) setCondensed(checked);
                    setShowPredictions(checked);
                  }}
                  disabled={arePredictionsLoading || isLoading}
                  className="data-[state=checked]:bg-[#4db8d8]"
                  id="advanced"
                />
                <Label htmlFor="advanced">Advanced View</Label>
              </div>
              {isAdvanced &&
                predictionHistory &&
                state.predictions[ticker][0] && (
                  <Select
                    value={state.views.predictions.model ?? "average"}
                    onValueChange={(value) => {
                      resetCursor();
                      dispatch({
                        type: actions.SET_PREDICTION_VIEW_MODEL,
                        payload: { model: value },
                      });
                    }}
                  >
                    <SelectTrigger className="w-[180px]">
                      <SelectValue placeholder="Filter by model" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectGroup>
                        <SelectLabel>Models</SelectLabel>
                        {/* FIXME: POTENTIAL CACHE MISMATCH (datahandler doesnt store predictions in global state yet) */}
                        {Object.keys(state.predictions[ticker][0]).map(
                          (key) => {
                            if (key === "day") return null;
                            return (
                              <SelectItem key={key} value={key}>
                                {key}
                              </SelectItem>
                            );
                          }
                        )}
                        <SelectSeparator />
                        <Button
                          className="w-full px-2"
                          variant="secondary"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            resetCursor();
                            dispatch({
                              type: actions.SET_PREDICTION_VIEW_MODEL,
                              payload: { model: "" },
                            });
                          }}
                        >
                          Clear
                        </Button>
                      </SelectGroup>
                    </SelectContent>
                  </Select>
                )}
              {isAdvanced && (
                <div className="flex items-center space-x-2">
                  <Switch
                    checked={isCondensed}
                    onCheckedChange={(checked) => {
                      setCondensed(checked);
                    }}
                    disabled={arePredictionsLoading || isLoading}
                    className="data-[state=checked]:bg-[#8e8e8e]"
                    id="condensed"
                  />
                  <Label htmlFor="condensed">Condense</Label>
                </div>
              )}
            </div>
            <div className="flex flex-col items-center sm:items-start justify-start">
              {/* model accuracy can only be evaluated based on stock close */}
              {chartData &&
                isAdvanced &&
                dataKeyInput === "stock_close" &&
                showPredictions && <HistoricalAccuracy data={chartData} />}
            </div>
          </CardContent>
        </Card>
      )}
    </>
  );
}
