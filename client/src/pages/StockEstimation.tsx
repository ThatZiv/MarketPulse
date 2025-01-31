import { useSupabase } from "@/database/SupabaseProvider";


export default function Stocks() {
  const {user} = useSupabase();
    return (
      <div className="p-4 w-7/12">
        <h1 className="font-semibold text-3xl pb-6">TESLA</h1>
        <div className="border border-black p-4 bg-tertiary/50 rounded-lg">
          <h2 className="font-semibold">Hey {user?.email ?? "Guest"},</h2>
          <h3>Current Stock Rate: $ 10.12</h3>
          <div className="flex flex-row justify-center gap-64 mt-4">
            <div className="flex flex-col">
            <h3 className="text-lg">Number of Stocks Invested:</h3>
            <p className="text-4xl">10</p>
            </div>
            <div className="flex flex-col">
            <h3 className="text-lg">Current Stock Earnings:</h3>
            <p className="text-4xl">$101.12</p>
            </div>
          </div>
        </div>
      </div>
    );
  }
  