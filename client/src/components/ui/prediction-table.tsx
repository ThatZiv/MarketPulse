import { useGlobal } from "@/lib/GlobalProvider";
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
import {
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import moment from "moment";
import { Button } from "@/components/ui/button";
import { capitalizeFirstLetter } from "@/lib/utils";
import { actions } from "@/lib/constants";
import InfoTooltip from "@/components/InfoTooltip";
import { PurchaseHistoryCalculator } from "@/lib/Calculator";

interface PredictionTableProps {
  ticker: string;
}

function ColoredRow(row: { row: (number | string)[]; value: number }) {
  let greatest = true;
  let least = true;

  row.row.forEach((x: number | string) => {
    if (!isNaN(Number(x))) {
      if (Number(x) > row.value) {
        greatest = false;
      }
      if (Number(x) < row.value) {
        least = false;
      }
    }
  });

  const valueStr = PurchaseHistoryCalculator.toDollar(row.value);
  return (
    <div className="flex justify-center items-center">
      <p
        className={`${
          greatest ? "text-green-700" : least ? "text-red-700" : ""
        } px-2`}
      >
        {valueStr}
      </p>
      {(greatest || least) && (
        <InfoTooltip side="right">
          {greatest
            ? "This is the prediction with the highest value on this day."
            : "This is the prediction with the lowest value on this day"}
        </InfoTooltip>
      )}
    </div>
  );
}

export default function PredictionTable({ ticker }: PredictionTableProps) {
  const { state, dispatch } = useGlobal();
  const { predictions } = state;
  const model = state.views.predictions.model;
  const days = state.views.predictions.timeWindow;

  const data = predictions[ticker];
  if (!data || data.length === 0) {
    return <div>Prediction table is currently unavailable</div>;
  }

  const setModel = (value: string) => {
    dispatch({
      type: actions.SET_PREDICTION_VIEW_MODEL,
      payload: { model: value },
    });
  };

  const setDays = (value: number) => {
    dispatch({
      type: actions.SET_PREDICTION_VIEW_TIME,
      payload: { timeWindow: value },
    });
  };

  return (
    <>
      <div className="flex w-full space-x-2 items-center justify-end">
        <Select value={model ?? ""} onValueChange={(value) => setModel(value)}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Filter by model" />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              <SelectLabel>Models</SelectLabel>
              {Object.keys(data[0]).map((key) => {
                if (key === "day") return null;
                return (
                  <SelectItem key={key} value={key}>
                    {key}
                  </SelectItem>
                );
              })}
              <SelectSeparator />
              <Button
                className="w-full px-2"
                variant="secondary"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  setModel("");
                }}
              >
                Clear
              </Button>
            </SelectGroup>
          </SelectContent>
        </Select>
        <Select
          value={days.toString()}
          onValueChange={(value) => setDays(Number(value))}
        >
          <SelectTrigger className="w-[180px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              <SelectLabel>Time windows</SelectLabel>
              <SelectItem value="1">1 Day</SelectItem>
              <SelectItem value="7">1 Week</SelectItem>
              <SelectSeparator />
              <Button
                className="w-full px-2"
                variant="secondary"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  setDays(7);
                }}
              >
                Clear
              </Button>
            </SelectGroup>
          </SelectContent>
        </Select>
      </div>
      <Table>
        <TableHeader>
          <TableRow className="font-bold">
            {Object.keys(data[0]).map((key) => {
              if (model && model !== key && key !== "day") return null;
              return (
                <TableCell key={key}>{capitalizeFirstLetter(key)}</TableCell>
              );
            })}
          </TableRow>
        </TableHeader>
        <TableBody>
          {data
            .map((row, i) => (
              <TableRow key={i}>
                {Object.values(row).map((value, j) => {
                  let valueStr = value.toString();
                  if (!isNaN(Number(value))) {
                    valueStr = "$" + Number(value).toFixed(2);
                    if (model && model !== Object.keys(data[0])[j]) return null;
                    else {
                      return (
                        <TableCell key={`${i}:${j}`}>
                          {j === Object.values(row).length - 1 ? (
                            // last column is for AVERAGE model!
                            <div className="flex justify-center items-center">
                              <p className="text-orange-600 px-2">{valueStr}</p>
                              <InfoTooltip side="right">
                                This is the average of the model predictions on
                                this day.
                              </InfoTooltip>
                            </div>
                          ) : (
                            <ColoredRow
                              row={Object.values(row)}
                              value={Number(value)}
                            />
                          )}
                        </TableCell>
                      );
                    }
                  } else if (moment(value).isValid()) {
                    valueStr = moment(valueStr).format("YYYY-MM-DD");
                  }
                  return <TableCell key={`${i}:${j}`}>{valueStr}</TableCell>;
                })}
              </TableRow>
            ))
            .slice(0, days)}
        </TableBody>
      </Table>
    </>
  );
}
