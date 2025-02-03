import { useState } from "react";
import { useSupabase } from "@/database/SupabaseProvider";
import { useNavigate } from "react-router-dom";
import { ArrowRight } from "lucide-react";
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

// Define the structure for stock form data
interface StockFormData {
  /**
   * this is the stock_id
   */
  ticker: string;
  hasStocks: string;
  sharesOwned: number;
  cashToInvest: number;
}

export default function StockPage() {
  const navigate = useNavigate();
  const { user, supabase } = useSupabase();

  // Manage form state
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
    hasStocks: z.string().nonempty(),
    sharesOwned: z.number().int().min(0).optional(),
    cashToInvest: z
      .number()
      .int()
      .min(1, "Cash to invest must be greater than 0"),
  });

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // use zod

    const { error } = formSchema.safeParse(formData);
    if (error) {
      for (const issue of error.issues) {
        setError(issue.message);
      }
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
      success: "Stock data saved successfully",
      error: (err) => `Failed to save stock data: ${err.message}`,
    });

    navigate("/");
  };

  // Handle input field changes
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
    <main>
      <header className="relative h-16 px-4 border-b border-gray-200 flex items-center justify-between">
        <h1 className="text-4xl font-[Poppins] font-bold text-center flex-1 tracking-tight">
          Stock Details
        </h1>
      </header>

      <main className="min-h-[calc(100vh-4rem)] dark:text-black text-left p-8 flex flex-col items-center justify-center">
        <form
          onSubmit={handleSubmit}
          className="bg-white w-[600px] rounded-lg p-8 shadow-md"
        >
          {error && (
            <div className="mb-4 text-red-500 text-center">{error}</div>
          )}

          {/* Stock Ticker Selection */}
          <div className="mb-6">
            <label htmlFor="ticker" className="block text-lg font-light mb-2">
              What is the ticker?
            </label>
            {stocksLoading ? (
              <Skeleton className="w-full h-12" />
            ) : (
              <Select
                value={formData.ticker}
                defaultValue={""}
                onValueChange={(value) => {
                  setFormData((prev) => ({ ...prev, ticker: value }));
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
              Do you already have stocks for this ticker?
            </label>
            <select
              id="hasStocks"
              className="w-full border border-gray-300 rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
              value={formData.hasStocks}
              onChange={handleInputChange}
              required
            >
              <option value="">Select Option</option>
              <option value="yes">Yes</option>
              <option value="no">No</option>
            </select>
          </div>

          {formData.hasStocks === "yes" && (
            <div className="mb-6">
              <label
                htmlFor="sharesOwned"
                className="block text-lg font-light mb-2"
              >
                How many stocks do you own?
              </label>
              <input
                id="sharesOwned"
                type="number"
                min="0"
                className="w-full border border-gray-300 rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
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
              How much cash do you want to invest in this stock? ($)
            </label>
            <input
              id="cashToInvest"
              type="number"
              min="0"
              step="0.01"
              className="w-full border border-gray-300 rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
              value={formData.cashToInvest}
              onChange={handleInputChange}
              required
            />
          </div>

          <div className="flex justify-between mt-8">
            <button
              type="button"
              className="bg-[#DFF6D7] px-6 py-3 rounded-full text-lg font-bold shadow-md transform hover:scale-105 active:scale-95 hover:bg-green-300 active:bg-green-400 transition-all duration-200"
              onClick={() => navigate("/")}
            >
              Return
            </button>
            <button
              type="submit"
              className="bg-[#DFF6D7] px-6 py-3 rounded-full text-lg font-bold shadow-md transform hover:scale-105 active:scale-95 flex justify-normal items-center hover:bg-green-300 active:bg-green-400 transition-all duration-200 disabled:opacity-50"
              disabled={stocksLoading}
            >
              Submit <ArrowRight />
            </button>
          </div>
        </form>
      </main>
    </main>
  );
}
