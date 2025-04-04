import { ForecastModelCalculator } from "@/lib/Calculator";
import { ChartDatapoint } from "@/types/global_state";
import React from "react";

interface HistoricalAccuracyProps {
  data: ChartDatapoint[];
}
export default function HistoricalAccuracy({
  data,
}: HistoricalAccuracyProps): JSX.Element {
  const { actual, predicted } = data.reduce(
    (acc, datapoint) => {
      let start = false;
      for (const [key, value] of Object.entries(datapoint)) {
        if (key === "date") {
          continue;
        } else if (key === "stock_close") {
          if (start) acc["actual"].push(Number(value));
        } else if (!start) {
          console.log(key);
          // this is for model output
          acc["predicted"].push(Number(value));
          start = true;
        }
      }
      return acc;
    },
    { actual: [], predicted: [] } as { actual: number[]; predicted: number[] }
  );
  console.log("actual", actual);
  console.log("predicted", predicted);

  const calc = new ForecastModelCalculator(actual, predicted);

  return (
    <div className="flex flex-col items-center justify-center w-full h-full p-4">
      {/* <h2 className="text-2xl font-bold">Model Performance</h2>
      <p className="mt-4 text-lg">
        Our historical accuracy is a measure of how well our predictions align
        with actual outcomes.
      </p> */}
      <div className="mt-8">{calc.accuracy(0.2)}</div>
    </div>
  );
}
