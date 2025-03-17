import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Label,
  LabelList,
  YAxis,
} from "recharts";

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { useQuery } from "@tanstack/react-query";
import { actions, cache_keys } from "@/lib/constants";
import { useSupabase } from "@/database/SupabaseProvider";
import { useMemo } from "react";
import { type PurchaseHistoryDatapoint } from "@/types/global_state";
import moment from "moment";
import { Skeleton } from "./ui/skeleton";
import { useGlobal } from "@/lib/GlobalProvider";
import { PurchaseHistoryCalculator } from "@/lib/Calculator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import { SelectSeparator } from "./ui/select";
// const chartData = [
//   { month: "January", visitors: 186 },
//   { month: "February", visitors: 205 },
//   { month: "March", visitors: -207 },
//   { month: "April", visitors: 173 },
//   { month: "May", visitors: -209 },
//   { month: "June", visitors: 214 },
// ];

interface PurchaseHistoryProps {
  ticker: string;
  stock_id: number;
}

const chartConfig = {
  amount_purchased: {
    label: "Shares",
  },
} satisfies ChartConfig;

export default function PurchaseHistory({
  ticker,
  stock_id,
}: PurchaseHistoryProps) {
  const { supabase, user } = useSupabase();
  const {
    // state: { history },
    dispatch,
  } = useGlobal();
  const {
    data: purchases,
    isLoading,
    isError,
  } = useQuery<Array<PurchaseHistoryDatapoint>>({
    queryKey: [cache_keys.USER_STOCK_TRANSACTION, ticker],
    queryFn: async () => {
      // TODO: take from global state (read-thru cache)
      const { data, error } = await supabase
        .from("User_Stock_Purchases")
        .select("*")
        .eq("stock_id", stock_id)
        .eq("user_id", user?.id)
        .order("date", { ascending: true });
      if (error) throw error;
      dispatch({
        type: actions.SET_USER_STOCK_TRANSACTIONS,
        payload: {
          stock_ticker: ticker,
          data,
        },
      });

      return data || [];
    },
    enabled: !!ticker,
  });

  const chartData = useMemo(() => {
    const points: Array<PurchaseHistoryDatapoint> = [];
    if (!purchases) return points;
    for (const { date, amount_purchased, price_purchased } of purchases) {
      points.push({
        date: moment(date).format("MMM DD"),
        amount_purchased,
        price_purchased,
      });
    }
    return points;
  }, [purchases]);

  const calc = useMemo(() => {
    return new PurchaseHistoryCalculator(purchases ?? []);
  }, [purchases]);
  return (
    <Card className="border border-black dark:border-white rounded-md md:p-4 mt-4">
      <CardHeader>
        <CardTitle>Your {ticker} Purchase History</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading && (
          <div className="flex items-end justify-end gap-2">
            <Skeleton className="h-40 w-full rounded-none" />
            <Skeleton className="h-16 w-full rounded-none" />
            <Skeleton className="h-24 w-full rounded-none" />
            <Skeleton className="h-12 w-full rounded-none" />
            <Skeleton className="h-32 w-full rounded-none" />
          </div>
        )}
        {purchases && purchases?.length > 0 ? (
          <>
            <CardDescription>
              Since {moment(purchases[0].date).calendar()}
            </CardDescription>
            <ChartContainer config={chartConfig} className="h-[300px] w-full">
              <BarChart accessibilityLayer data={chartData}>
                <CartesianGrid vertical={false} />
                <ChartTooltip
                  cursor={false}
                  content={<ChartTooltipContent hideLabel hideIndicator />}
                />
                <YAxis
                  dataKey="amount_purchased"
                  label={
                    <Label value="Shares" angle={-90} position="insideLeft" />
                  }
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(value) => {
                    return value > 0 ? `+${value}` : value;
                  }}
                />
                <Bar dataKey="amount_purchased" name="Shares">
                  <LabelList
                    className="text-sm sm:text-md"
                    position="bottom"
                    dataKey="date"
                    fillOpacity={3}
                  />
                  {chartData.map((item) => (
                    <Cell
                      key={item.date}
                      fill={
                        Number(item.amount_purchased) > 0
                          ? "#16C47F"
                          : "#F93827"
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ChartContainer>
            <SelectSeparator />
            <Table className="w-full">
              <TableHeader>
                <TableRow>
                  <TableHead>Date</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Shares</TableHead>
                  <TableHead>Price</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody className="text-left">
                {purchases.map((purchase) => (
                  <TableRow key={purchase.date}>
                    <TableCell>
                      {moment(purchase.date).format("MMMM DD, yyyy")}
                    </TableCell>
                    <TableCell>
                      {purchase.amount_purchased > 0 ? "Buy" : "Sell"}
                    </TableCell>
                    <TableCell>{Math.abs(purchase.amount_purchased)}</TableCell>
                    <TableCell>
                      {PurchaseHistoryCalculator.toDollar(
                        purchase.price_purchased
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </>
        ) : (
          <h1>No purchase history available</h1>
        )}
      </CardContent>
      <CardFooter className="flex-col items-start gap-2 text-sm">
        {isError ? (
          <div className="leading-none text-muted-foreground">
            <div>Error loading purchase history</div>
          </div>
        ) : (
          purchases &&
          purchases?.length > 0 && (
            <div className="flex gap-1 font-medium leading-none">
              <span
                className={
                  calc.getProfit() > 0 ? "text-green-600" : "text-red-600"
                }
              >
                {PurchaseHistoryCalculator.toDollar(calc.getProfit())}
              </span>
              was {calc.getProfit() > 0 ? "made" : "lost"} from your last sale.
            </div>
          )
        )}
      </CardFooter>
    </Card>
  );
}
