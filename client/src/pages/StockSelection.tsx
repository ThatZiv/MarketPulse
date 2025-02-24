import { useState } from "react";
import { useSupabase } from "@/database/SupabaseProvider";
import { useNavigate } from "react-router-dom";
import { ArrowRight, ArrowLeft } from "lucide-react";
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

interface StockFormData {
  ticker: string;
  hasStocks: string;
  sharesOwned: number;
  cashToInvest: number;
}

export default function StockPage() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { user, supabase } = useSupabase();

  const [formData, setFormData] = useState<StockFormData>({
    ticker: "",
    hasStocks: "",
    sharesOwned: 0,
    cashToInvest: 0,
  });
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
  const formSchema = z.object({
    ticker: z.string().nonempty("Please select a stock"),
    hasStocks: z
      .string()
      .nonempty("Please specify if you own shares for this stock"),
    sharesOwned: z.number().min(0).optional(),
    cashToInvest: z.number().min(1, "Cash to invest must be greater than 0"),
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const { error } = formSchema.safeParse(formData);
    if (error) {
      error.errors.reverse().forEach((err) => setError(err.message));
      return;
    }

    const updateStock = new Promise((resolve, reject) => {
      supabase
        .from("User_Stocks")
        .upsert(
          {
            user_id: user?.id,
            stock_id: formData.ticker,
            shares_owned:
              formData.hasStocks === "yes" ? formData.sharesOwned : 0,
            desired_investiture: formData.cashToInvest,
          },
          { onConflict: "user_id,stock_id" }
        )
        .then(({ error }) => {
          if (error) reject(error);
          resolve(null);
        });
    });

    toast.promise(updateStock, {
      loading: "Saving...",
      success: async () => {
        await queryClient.invalidateQueries({
          queryKey: [cache_keys.USER_STOCKS],
        });
        return "Stock data saved successfully";
      },
      error: (err) => `Failed to save stock data: ${err.message}`,
    });
    navigate("/");
  };

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { id, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [id]:
        id === "sharesOwned" || id === "cashToInvest" ? Number(value) : value,
    }));
  };

  if (stocksError) {
    return (
      <div className="text-center flex items-center">
        <h1 className="text-3xl">Error</h1>
        <p className="text-primary">
          Unfortunately, we encountered an error fetching the stocks:{" "}
          {(stocksError as Error).message}
        </p>
      </div>
    );
  }
  return (
    <main className="w-xl min-h-screen">
      <header className="px-4 border-b flex items-center justify-between mx-auto max-w-screen-sm">
        <h1 className="text-4xl mb-2 text-center flex-1 tracking-tight">
          Stock Details
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
                defaultValue={""}
                onValueChange={(value: string) => {
                  setFormData((prev) => ({ ...prev, ticker: value }));
                }}
                required
              >
                <SelectTrigger>
                  <SelectValue
                    placeholder="Select Stock"
                    className="dark:text-white"
                  />
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
              Do you already own stocks for this ticker?{" "}
              <span className="text-red-500">*</span>
            </label>
            <Select
              value={formData.hasStocks}
              onValueChange={(value: string) =>
                setFormData((prev) => ({ ...prev, hasStocks: value }))
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
              <label
                htmlFor="sharesOwned"
                className="block text-lg font-light mb-2"
              >
                How many shares do you own?{" "}
                <span className="text-red-500">*</span>
              </label>
              <input
                id="sharesOwned"
                type="number"
                step="any"
                min="0"
                className="w-full border border-gray-300 bg-white text-black dark:text-white dark:bg-black rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
                value={formData.sharesOwned}
                onChange={handleInputChange}
                required
              />
            </div>
          )}

          <div className="mb-6">
            <label
              htmlFor="cashToInvest"
              className="block text-lg font-light mb-2"
            >
              How much cash do you want to invest in this stock?{" "}
              <span className="text-red-500">*</span>
            </label>
            <input
              id="cashToInvest"
              type="number"
              min="0"
              step="any"
              className="w-full bg-white dark:bg-black dark:text-white border ring-offset-background rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
              value={formData.cashToInvest}
              onChange={handleInputChange}
              required
            />
          </div>

          <div className="flex flex-col md:flex-row gap-3 md:gap-0 justify-between mt-8">
            <Button
              type="button"
              className="flex items-center justify-center hover:bg-primary/60 active:bg-primary/70 disabled:opacity-50 w-full sm:w-auto"
              onClick={() => navigate("/")}
            >
              <ArrowLeft className="hidden md:block" />
              Back
            </Button>
            <Button
              type="submit"
              className="flex items-center justify-center hover:bg-primary/60 active:bg-primary/70 disabled:opacity-50 w-full sm:w-auto"
              disabled={stocksLoading}
            >
              Submit <ArrowRight className="hidden md:block" />
            </Button>
          </div>
        </form>
      </main>
    </main>
  );
}
