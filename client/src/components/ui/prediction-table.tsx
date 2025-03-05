import { useGlobal } from "@/lib/GlobalProvider";
import React from "react";
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

export default function PredictionTable({ ticker }: PredictionTableProps) {
  const { state, dispatch } = useGlobal();
  const { predictions } = state;
  const model = state.views.predictions.model;
  const days = state.views.predictions.timeWindow;

  const data = predictions[ticker];
  if (!data || data.length === 0) {
    return <div>No data available</div>;
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
