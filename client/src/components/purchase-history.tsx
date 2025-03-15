import { TrendingDown, TrendingUp } from "lucide-react";
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
        date: moment(date).calendar(),
        amount_purchased,
        price_purchased,
      });
    }
    return points;
  }, [purchases]);

  const currentValue = useMemo(() => {
    if (!purchases || purchases.length === 0) return 0;
    return purchases.reduce((acc, { amount_purchased, price_purchased }) => {
      return acc + amount_purchased * price_purchased;
    }, 0);
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
            <ChartContainer config={chartConfig}>
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
                  <LabelList position="top" dataKey="date" fillOpacity={1} />
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
          <div className="flex gap-2 font-medium leading-none">
            ${currentValue}{" "}
            {currentValue > 0 ? (
              <TrendingUp className="h-4 w-4" />
            ) : (
              <TrendingDown className="h-4 w-4" />
            )}
          </div>
        )}
      </CardFooter>
    </Card>
  );
}
