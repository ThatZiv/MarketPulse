import { ForecastModelCalculator } from "@/lib/Calculator";
import { expandDomain } from "@/lib/utils";
import { ChartDatapoint } from "@/types/global_state";
import { Separator } from "@/components/ui/separator";
import InfoTooltip from "./InfoTooltip";

interface HistoricalAccuracyProps {
  data: ChartDatapoint[];
}
export default function HistoricalAccuracy({
  data,
}: HistoricalAccuracyProps): JSX.Element {
  let actual: number[] = [];
  const predicted: number[] = [];
  // data must have date, stock_close, and *_prediction
  if (data && expandDomain(data).length === 3) {
    for (const item of data) {
      for (const [key, value] of Object.entries(item)) {
        if (key.endsWith("_prediction")) {
          predicted.push(value as number);
        } else if (key === "stock_close") {
          actual.push(value as number);
        }
      }
    }
  }
  // make sure actual and predicted are the same length
  actual = actual.slice(0, predicted.length);

  const calc = new ForecastModelCalculator(actual, predicted);
  const metrics = [
    {
      label: "Accuracy",
      value: (calc.accuracy() * 100).toFixed(2),
      unit: "%",
      explanation:
        "Accuracy is the percentage of predictions that are within a certain threshold of the actual value. The threshold is set to 25% of the range (minimum to maximum) of the actual values.",
    },
    {
      label: "MAE",
      value: calc.meanAbsoluteError().toFixed(2),
      unit: "",
      explanation:
        "Mean Absolute Error (MAE) is the average of the absolute errors between predicted and actual values. It gives an idea of how far off the predictions are from the actual values.",
    },
    {
      label: "MSE",
      value: calc.meanSquaredError().toFixed(2),
      unit: "",
      explanation:
        "Mean Squared Error (MSE) is the average of the squared differences between predicted and actual values. It gives more weight to larger errors.",
    },
    {
      label: "RMSE",
      value: calc.rootMeanSquaredError().toFixed(2),
      unit: "",
      explanation:
        "Root Mean Squared Error (RMSE) is the square root of the average of the squared differences between predicted and actual values. It gives an idea of how far off the predictions are from the actual values.",
    },
    {
      label: "R²",
      value: calc.rSquared().toFixed(2),
      unit: "",
      explanation:
        "R-squared (R²) is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model.",
    },
    {
      label: "MAPE",
      value: calc.meanAbsolutePercentageError().toFixed(2),
      unit: "%",
      explanation:
        "Mean Absolute Percentage Error (MAPE) is the average of the absolute percentage errors between predicted and actual values. It gives an idea of how far off the predictions are from the actual values.",
    },
  ];
  return (
    <div className="flex flex-col items-center justify-center w-full">
      {/* <h2 className="text-2xl font-bold">Model Performance</h2>
      <p className="mt-4 text-lg">
        Our historical accuracy is a measure of how well our predictions align
        with actual outcomes.
      </p> */}
      <Separator className="my-4" />

      <h4 className="text-sm font-medium leading-none mb-2">
        Model Performance
      </h4>
      <p className="text-sm text-muted-foreground">
        Our historical accuracy is a measure of how well our predictions align
        with actual outcomes.
      </p>
      <Separator className="my-2" />
      <div className="flex h-5 items-center space-x-4 mt-2 text-sm flex-wrap justify-center">
        {metrics.map(({ label, value, unit, explanation }, index) => (
          <>
            <div key={index} className="flex items-center">
              <InfoTooltip size="md" className="mr-1">
                {explanation}
              </InfoTooltip>
              <span className="font-medium flex">
                {label}: {value} {unit}
              </span>
            </div>
            {index < metrics.length - 1 && (
              <Separator key={index + "vert"} orientation="vertical" />
            )}
          </>
        ))}
      </div>
    </div>
  );
}
