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

interface PredictionTableProps {
  ticker: string;
}
function Average(row: { row: (number | string)[] }) {
  let average = 0;
  let count = 0;
  //console.log(row.row[1]);

  row.row.forEach((x: number | string) => {
    if (!isNaN(Number(x))) {
      average += Number(x);
      count += 1;
    }
  });
  //console.log(average);
  average = average / count;

  const output = "$" + Number(average).toFixed(2);
  return <TableCell>{output}</TableCell>;
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
  //console.log(average);

  const valueStr = "$" + Number(row.value).toFixed(2);
  if (greatest) {
    return <p className="text-green-700">{valueStr}</p>;
  } else if (least) {
    return <p className="text-red-700">{valueStr}</p>;
  } else {
    return <p>{valueStr}</p>;
  }
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
            {model === "" ? <TableCell>Average</TableCell> : <></>}
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
                          <ColoredRow
                            row={Object.values(row)}
                            value={Number(value)}
                          />
                        </TableCell>
                      );
                    }
                  } else if (moment(value).isValid()) {
                    valueStr = moment(valueStr).format("YYYY-MM-DD");
                  }
                  return <TableCell key={`${i}:${j}`}>{valueStr}</TableCell>;
                })}
                {model === "" ? <Average row={Object.values(row)} /> : <></>}
              </TableRow>
            ))
            .slice(0, days)}
        </TableBody>
      </Table>
    </>
  );
}
