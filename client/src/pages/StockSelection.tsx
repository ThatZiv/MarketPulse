import { useEffect, useMemo, useState } from "react";
import { useSupabase } from "@/database/SupabaseProvider";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  ArrowRight,
  ArrowLeft,
  ArrowDown,
  ArrowUp,
  TrendingUp,
  Box,
  Trash,
  Plus,
  Undo,
  X,
} from "lucide-react";
import useAsync from "@/hooks/useAsync";
import { type Stock } from "@/types/stocks";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { useQueryClient } from "@tanstack/react-query";
import { cache_keys } from "@/lib/constants";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Separator } from "@/components/ui/separator";
import { PurchaseHistoryCalculator } from "@/lib/Calculator";
import InfoTooltip from "@/components/InfoTooltip";

const getTodayISOString = () => {
  const today = new Date();
  const year = today.getFullYear();
  const month = String(today.getMonth() + 1).padStart(2, "0");
  const day = String(today.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
};

export interface StockFormData {
  ticker: string;
  hasStocks: string;
  purchases: {
    date: string;
    shares: number | null;
    pricePurchased: number | null;
  }[];
  cashToInvest: number | null;
}

export default function StockPage() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { user, supabase } = useSupabase();

  const title = useMemo(
    () => (searchParams.has("ticker") ? "Edit Stock" : "Add New Stock"),
    [searchParams]
  );

  const [formData, setFormData] = useState<StockFormData>({
    ticker: "",
    hasStocks: "",
    purchases: [],
    cashToInvest: null,
  });

  const [previousPurchases, setPreviousPurchases] = useState<
    StockFormData["purchases"]
  >([]);

  const [error, setError] = useState<string>();
  const {
    value: stocks,
    error: stocksError,
    loading: stocksLoading,
  } = useAsync<Stock[]>(
    () =>
      new Promise((resolve, reject) => {
        supabase
          .from("Stocks")
          .select("*")
          .then(({ data, error }) => {
            if (error) reject(error);
            return resolve(data || []);
          });
      }),
    [supabase]
  );

  useEffect(() => {
    if (searchParams.has("ticker") && stocks) {
      const ticker = (searchParams.get("ticker") as string).toUpperCase();
      const stock = stocks.find((stock) => stock.stock_ticker === ticker);
      if (stock) {
        setFormData((prev) => ({ ...prev, ticker: stock.stock_id.toString() }));
        fetchPurchaseHistory(stock.stock_id.toString());
      }
    }
  }, [searchParams, stocks]);

  const formSchema = z
    .object({
      ticker: z.string().nonempty("Please select a stock"),
      hasStocks: z
        .string()
        .nonempty("Please specify if you own shares for this stock"),
      purchases: z.array(
        z.object({
          date: z
            .string()
            .nonempty("Purchase date is required")
            .refine((date) => {
              const selectedDate = new Date(date);
              const today = new Date();
              today.setHours(23, 59, 59, 999);
              const minDate = new Date(2000, 0, 1);
              return selectedDate >= minDate && selectedDate <= today;
            }, "Date must be between January 1, 2000 and today"),
          shares: z.number(),
          pricePurchased: z.number().min(0.01, "Price must be at least $0.01"),
        })
      ),
      cashToInvest: z.number().min(1, "Cash to invest must be greater than 0"),
    })
    .refine(
      (data) => {
        if (data.hasStocks === "yes") {
          return data.purchases.length > 0;
        }
        return true;
      },
      {
        message: "At least one purchase entry is required if you own stocks",
        path: ["purchases"],
      }
    );

  const addPurchaseEntry = () => {
    setFormData((prev) => ({
      ...prev,
      purchases: [
        ...prev.purchases,
        { date: "", shares: null, pricePurchased: null },
      ],
    }));
  };

  const removePurchaseEntry = (index: number) => {
    setFormData((prev) => ({
      ...prev,
      purchases: prev.purchases.filter((_, i) => i !== index),
    }));
  };

  const resetPurchaseEntries = () => {
    setFormData((prev) => ({
      ...prev,
      purchases: previousPurchases,
    }));
  };

  const calc = useMemo(() => {
    return new PurchaseHistoryCalculator(
      // must convert to form data to db schema
      formData.purchases.map((purchase) => ({
        date: purchase.date,
        amount_purchased: purchase.shares ?? 0,
        price_purchased: purchase.pricePurchased ?? 0,
      }))
    );
  }, [formData.purchases]);

  const handlePurchaseChange = (
    index: number,
    field: "date" | "shares" | "pricePurchased" | "type",
    value: string
  ) => {
    const newPurchases = [...formData.purchases];
    if (field === "date") {
      // ignore if date is already in use (PK unique con)
      if (newPurchases.some((purchase) => purchase.date === value)) {
        toast.error("Date already in use");
        return;
      }
    }
    newPurchases[index] = {
      ...newPurchases[index],
      [field]:
        field === "shares" || field === "pricePurchased"
          ? value === ""
            ? null
            : Number(value)
          : value,
    };
    newPurchases.sort(
      (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
    );
    setFormData((prev) => ({ ...prev, purchases: newPurchases }));
  };

  const fetchPurchaseHistory = async (ticker: string) => {
    if (!user?.id) return;

    const { data, error } = await supabase
      .from("User_Stock_Purchases")
      .select("date, amount_purchased, price_purchased")
      .order("date", { ascending: true })
      .eq("user_id", user.id)
      .eq("stock_id", ticker);

    if (error) {
      console.error("Error fetching purchase history:", error);
      return;
    }
    const purchases = data.map((purchase) => ({
      date: purchase.date.split("T")[0],
      shares: purchase.amount_purchased,
      pricePurchased: purchase.price_purchased,
    }));
    setFormData((prev) => ({
      ...prev,
      ticker,
      hasStocks: data.length > 0 ? "yes" : "no",
      purchases,
    }));

    setPreviousPurchases(purchases);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const result = formSchema.safeParse(formData);
    if (!result.success) {
      result.error.errors.reverse().forEach((err) => setError(err.message));
      return;
    }

    const badDay = calc.isInvalidHistory();
    if (badDay) {
      setError(`You cannot sell more shares than you own on ${badDay}`);
      return;
    }

    const updateStock = new Promise((resolve, reject) => {
      supabase
        .from("User_Stocks")
        .upsert(
          {
            user_id: user?.id,
            stock_id: formData.ticker,
            desired_investiture: formData.cashToInvest,
          },
          { onConflict: "user_id,stock_id" }
        )
        .then(({ error: userStockError }) => {
          if (userStockError) {
            reject(userStockError);
            return;
          }

          if (
            !(formData.hasStocks === "yes" && formData.purchases.length > 0)
          ) {
            resolve(null);
            return;
          }

          const purchasesData = formData.purchases.map((purchase) => ({
            user_id: user?.id,
            stock_id: formData.ticker,
            date: purchase.date,
            amount_purchased: purchase.shares,
            price_purchased: purchase.pricePurchased,
          }));

          //duplicate check
          const badDay = calc.isInvalidHistory();
          if (badDay) {
            reject(
              new Error(`You cannot sell more shares than you own on ${badDay}`)
            );
            return;
          }

          supabase
            .from("User_Stock_Purchases")
            .delete()
            .eq("user_id", user?.id)
            .eq("stock_id", formData.ticker)
            .then(({ error: deleteError }) => {
              if (deleteError) {
                reject(deleteError);
                return;
              }

              supabase
                .from("User_Stock_Purchases")
                .insert(purchasesData)
                .then(({ error: insertError }) => {
                  if (insertError) reject(insertError);
                  else resolve(null);
                });
            });
        });
    });

    toast.promise(updateStock, {
      loading: "Saving...",
      success: async () => {
        await queryClient.invalidateQueries({
          queryKey: [cache_keys.USER_STOCKS],
        });
        await navigate("/");
        return "Stock data saved successfully";
      },
      error: (err) => `Failed to save stock data: ${err.message}`,
    });
  };

  if (stocksError) {
    return (
      <div className="text-center flex items-center">
        <h1 className="text-3xl">Error</h1>
        <p className="text-primary">
          Error fetching stocks: {(stocksError as Error).message}
        </p>
      </div>
    );
  }

  return (
    <main className="w-xl min-h-screen">
      <header className="px-4 border-b flex items-center justify-between mx-auto max-w-screen-sm">
        <h1 className="text-4xl mb-2 text-center flex-1 tracking-tight">
          {title}
        </h1>
      </header>

      <main className="text-black dark:text-white text-left p-2 flex flex-col">
        <form
          onSubmit={handleSubmit}
          className="bg-white w-full rounded-lg p-8 shadow-md tex-center dark:bg-black"
        >
          {error && (
            <div className="mb-4 text-red-500 text-center">{error}</div>
          )}

          <div className="mb-6">
            <label htmlFor="ticker" className="block text-lg font-light mb-2">
              What is the ticker? <span className="text-red-500">*</span>
            </label>

            {stocksLoading ? (
              <Skeleton className="w-full h-12" />
            ) : (
              <Select
                value={formData.ticker}
                onValueChange={async (value: string) => {
                  await fetchPurchaseHistory(value);
                }}
                required
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select Stock" />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectLabel>Stocks</SelectLabel>
                    {stocks?.map(({ stock_id, stock_name, stock_ticker }) => (
                      <SelectItem key={stock_id} value={stock_id.toString()}>
                        {stock_name} ({stock_ticker})
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            )}
          </div>

          <div className="mb-6">
            <label
              htmlFor="hasStocks"
              className="block text-lg font-light mb-2"
            >
              Do you own this stock? <span className="text-red-500">*</span>
            </label>
            <Select
              value={formData.hasStocks}
              onValueChange={(value: string) =>
                setFormData((prev) => ({
                  ...prev,
                  hasStocks: value,
                  purchases: value === "no" ? [] : prev.purchases,
                }))
              }
              required
            >
              <SelectTrigger>
                <SelectValue placeholder="Select Option" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectItem value="yes">Yes</SelectItem>
                  <SelectItem value="no">No</SelectItem>
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>

          {formData.hasStocks === "yes" && (
            <div className="mb-6">
              <label className="block text-lg font-light">
                Purchase History <span className="text-red-500">*</span>
              </label>
              <p className="mb-2 text-gray-600 text-muted-foreground font-medium">Note: Please enter your stock history in the correct time sequence.</p>
              {formData.purchases.map((purchase, index) => (
                <div key={index} className="flex gap-2 mb-2">
                  <div className="flex-1 flex flex-col">
                    {index === 0 && (
                      <label htmlFor={`date-${index}`} className="text-sm mb-1">
                        Date
                      </label>
                    )}
                    <input
                      id={`date-${index}`}
                      type="date"
                      required
                      value={purchase.date}
                      min="2000-01-01"
                      max={getTodayISOString()}
                      onChange={(e) =>
                        handlePurchaseChange(index, "date", e.target.value)
                      }
                      className="w-full border border-gray-300 bg-white text-black dark:text-white dark:bg-black rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                  </div>
                  <div id={`type-${index}`} className="flex-1 flex flex-col">
                    {index === 0 && (
                      <label htmlFor={`type-${index}`} className="text-sm mb-1">
                        Type
                      </label>
                    )}
                    <Select
                      disabled={purchase.shares === null}
                      value={
                        purchase.shares !== null
                          ? purchase.shares > 0
                            ? "buy"
                            : "sell"
                          : "buy"
                      }
                      onValueChange={(value: string) => {
                        if (purchase.shares === null) return;

                        handlePurchaseChange(
                          index,
                          "shares",
                          String(
                            value == "buy"
                              ? Math.abs(purchase.shares)
                              : Math.abs(purchase.shares) * -1
                          )
                        );

                        // handlePurchaseChange(
                        //   index,
                        //   "shares",
                        //   String(value === "sell" && purchase.shares * -1)
                        // );
                      }}
                      //  onValueChange={(value: string) =>
                      required
                    >
                      <SelectTrigger className="border border-gray-300 bg-white text-black dark:text-white dark:bg-black rounded px-4 py-2 h-full focus:outline-none focus:ring-2 focus:ring-primary">
                        <SelectValue placeholder="Select Option" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectGroup>
                          <SelectItem value="buy">Buy</SelectItem>
                          <SelectItem value="sell">Sell</SelectItem>
                        </SelectGroup>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex-1 flex flex-col">
                    {index === 0 && (
                      <label
                        htmlFor={`shares-${index}`}
                        className="text-sm mb-1"
                      >
                        Shares
                      </label>
                    )}
                    <input
                      id={`shares-${index}`}
                      type="number"
                      step="0.01"
                      required
                      value={
                        purchase.shares === null
                          ? ""
                          : Math.abs(purchase.shares)
                      }
                      onChange={(e) =>
                        handlePurchaseChange(index, "shares", e.target.value)
                      }
                      className="w-full border border-gray-300 bg-white text-black dark:text-white dark:bg-black rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                  </div>

                  <div className="flex-1 flex flex-col">
                    {index === 0 && (
                      <label
                        htmlFor={`price-${index}`}
                        className="text-sm mb-1"
                      >
                        Price ($)
                      </label>
                    )}
                    <div className="flex justify-center items-center">
                      <input
                        id={`price-${index}`}
                        type="number"
                        step="0.01"
                        min="0.01"
                        required
                        value={purchase.pricePurchased ?? ""}
                        onChange={(e) =>
                          handlePurchaseChange(
                            index,
                            "pricePurchased",
                            e.target.value
                          )
                        }
                        className="w-full  border border-gray-300 bg-white text-black dark:text-white dark:bg-black rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
                      />
                    </div>
                  </div>

                  <Button
                    type="button"
                    onClick={() => removePurchaseEntry(index)}
                    variant="destructive"
                    className="self-end mb-1"
                  >
                    <Trash className="h-4 w-4" />
                  </Button>
                </div>
              ))}
              <div className="flex justify-between">
                <Button
                  type="button"
                  onClick={addPurchaseEntry}
                  className="mt-2"
                >
                  Add Purchase <Plus className="h-4 w-4" />
                </Button>
                {previousPurchases != formData.purchases && (
                  <Button type="button" onClick={resetPurchaseEntries}>
                    <Undo className="h-4 w-4" />
                    Revert Changes
                  </Button>
                )}
              </div>
              <Separator className="my-2" />
              <Table>
                <TableHeader>
                  {/* <TableCaption className="w-full">Totals</TableCaption> */}
                  <TableRow>
                    <TableHead>
                      <span className="flex justify-start items-center">
                        Current Profit <TrendingUp className="ml-2 h-4 w-4" />
                      </span>

                      <span className="text-xs">Based on last sale</span>
                    </TableHead>
                    <TableHead>
                      <span className="flex justify-start items-center">
                        Total Purchased
                        <ArrowDown className="ml-2 h-4 w-4" />
                      </span>
                    </TableHead>
                    <TableHead>
                      <span className="flex justify-start items-center">
                        Total Sold
                        <ArrowUp className="ml-2 h-4 w-4" />
                      </span>
                    </TableHead>
                    <TableHead>
                      <span className="flex justify-start items-center">
                        Current Shares
                        <Box className="ml-2 h-4 w-4" />
                      </span>
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow>
                    <TableCell>
                      <span
                        className={`${calc.getProfit() > 0
                          ? // totals.value * -1 is the value of profit
                          "text-green-600"
                          : "text-red-600"
                          }`}
                      >
                        {PurchaseHistoryCalculator.toDollar(calc.getProfit())}
                      </span>
                    </TableCell>
                    <TableCell>
                      {PurchaseHistoryCalculator.toDollar(
                        calc.getTotalBought()
                      )}
                    </TableCell>
                    <TableCell>
                      {PurchaseHistoryCalculator.toDollar(calc.getTotalSold())}
                    </TableCell>
                    <TableCell
                      className={`${calc.getTotalShares() < 0 ? "text-red-600" : ""
                        } flex items-center gap-2`}
                    >
                      {calc.getTotalShares().toLocaleString(undefined, {
                        maximumFractionDigits: 2,
                      })}
                      {calc.getTotalShares() < 0 && (
                        <InfoTooltip Icon={X} size="md">
                          You cannot sell more shares than you own.
                        </InfoTooltip>
                      )}
                    </TableCell>
                  </TableRow>
                </TableBody>
              </Table>
              { }
            </div>
          )}

          <div className="mb-6">
            <label
              htmlFor="cashToInvest"
              className="block text-lg font-light mb-2"
            >
              Desired Investment ($) <span className="text-red-500">*</span>
            </label>
            <input
              id="cashToInvest"
              type="number"
              min="0"
              step="0.01"
              className="w-full bg-white dark:bg-black dark:text-white border ring-offset-background rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
              value={formData.cashToInvest ?? ""}
              onChange={(e) =>
                setFormData((prev) => ({
                  ...prev,
                  cashToInvest:
                    e.target.value === "" ? null : Number(e.target.value),
                }))
              }
              required
            />
          </div>

          <div className="flex flex-col md:flex-row gap-3 md:gap-0 justify-between mt-8">
            <Button
              type="button"
              variant="outline"
              onClick={() => navigate("/")}
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>
            <Button type="submit" disabled={stocksLoading}>
              Submit
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </form>
      </main>
    </main>
  );
}
