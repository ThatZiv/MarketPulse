import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useSupabase } from "@/database/SupabaseProvider";
import useAsync from "@/hooks/useAsync";
import { Link } from "react-router";

interface StockResponse {
  Stocks: {
    stock_name: string;
    stock_ticker: string;
  };
  shares_owned: number;
}

interface StockCardProps {
  stock: StockResponse;
}

export default function Landing() {
  const { supabase, displayName, user } = useSupabase();

  const {
    value: stocks,
    error: stocksError,
    loading: loading,
  } = useAsync<StockResponse[]>(
    () =>
      new Promise((resolve, reject) => {
        supabase
          .from("User_Stocks")
          .select("Stocks (stock_name, stock_ticker), shares_owned")
          .eq("user_id", user?.id)
          .order("created_at", { ascending: false })
          .limit(5)
          .then(({ data, error }) => {
            if (error) reject(error);
            // @ts-expect-error Stocks will never expand to an array
            resolve(data || []);
          });
      }),
    [user, supabase]
  );

  if (stocksError) {
    return (
      <div className="flex flex-col justify-center items-center h-screen">
        <h1 className="text-3xl">Error</h1>
        <p className="text-primary">
          Unfortunately, we encountered an error fetching your stocks. Please
          refresh the page or try again later.
        </p>
      </div>
    );
  }
  return (
    <div className="min-h-screen">
      <h1 className="text-4xl text-center flex-1 tracking-tight">
        Welcome <b>{displayName || "User"}</b>
      </h1>
      <Separator className="my-2" />

      <div className="flex flex-col items-center gap-4 flex-grow">
        <section className="w-full">
          <h2 className="text-2xl font-light mb-6 text-center">
            Your Investment Portfolio
          </h2>
          {stocks?.length === 0 && (
            <div className="text-center text-gray-500 mb-4">
              No investments found, click the "+" to add your first investment
            </div>
          )}
          <div className="flex flex-row items-center justify-center gap-6">
            {loading ? (
              <>
                <Skeleton className="w-40 h-[100px]" />
                <Skeleton className="w-20 h-[100px]" />
                <Skeleton className="w-32 h-[100px]" />
              </>
            ) : (
              <div className="flex flex-row items-center justify-center gap-6">
                {stocks?.map((stock) => (
                  <StockCard key={stock?.Stocks?.stock_name} stock={stock} />
                ))}
              </div>
            )}
            <Link
              className="flex items-center justify-center h-20 w-20 text-white rounded-full pb-1 bg-primary text-4xl font-bold shadow hover:shadow-md transition-transform transform hover:scale-105 active:scale-95"
              to="/stocks"
            >
              +
            </Link>
          </div>
        </section>
      </div>
    </div>
  );
}

function StockCard({ stock }: StockCardProps) {
  return (
    <Link to={`/stocks/${stock.Stocks.stock_ticker}`}>
      <div className="bg-primary transition-all hover:px-8 p-6 rounded-lg text-white shadow flex flex-col justify-center items-center text-center hover:shadow-md">
        <h3 className="text-lg font-bold uppercase tracking-wide mb-4">
          {stock.Stocks.stock_name}
        </h3>
        <p className="text-sm font-medium">
          <span className="font-extrabold">
            {stock.shares_owned.toLocaleString()}
          </span>{" "}
          shares owned
        </p>
      </div>
    </Link>
  );
}
