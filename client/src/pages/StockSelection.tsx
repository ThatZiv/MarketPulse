import { useState } from "react";
import { useSupabase } from "@/database/SupabaseProvider";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { Separator } from "@/components/ui/separator";
import { useNavigate } from "react-router-dom";
import { Settings } from "lucide-react";

interface StockFormData {
  ticker: string;
  hasStocks: string;
  sharesOwned: number;
  cashToInvest: number;
}

export default function Stock() {
  const navigate = useNavigate();
  const { user, supabase } = useSupabase();
  
  const [formData, setFormData] = useState<StockFormData>({
    ticker: '',
    hasStocks: '',
    sharesOwned: 0,
    cashToInvest: 0
  });
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    if (!user) {
      setError('User not authenticated');
      setLoading(false);
      return;
    }

    try {
      const { error } = await supabase
        .from('User_Stocks')
        .insert([
          {
            user_id: user.id,
            stock_id: formData.ticker,
            shares_owned: formData.hasStocks === 'yes' ? formData.sharesOwned : 0,
            desired_investiture: formData.cashToInvest
          }
        ]);

      if (error) throw error;
      navigate("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save stock data');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { id, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [id]: id === 'sharesOwned' || id === 'cashToInvest' ? Number(value) : value
    }));
  };

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="relative h-16 bg-[#F5F5F5] px-4 border-b border-gray-200 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <SidebarTrigger className="h-6 w-6 text-black" />
            <Separator orientation="vertical" className="h-4" />
          </div>
          <h1 className="text-4xl font-[Poppins] font-bold text-center flex-1 tracking-tight text-gray-800">
            Stock Details:
          </h1>
          <button
            className="flex items-center justify-center p-2 rounded-full transform hover:scale-105 active:scale-95 hover:bg-gray-100 active:bg-gray-200 transition-all duration-200"
            onClick={() => navigate("/settings")}
          >
            <Settings className="h-6 w-6 text-gray-800" />
          </button>
        </header>

        <main className="bg-[#F5F5F5] min-h-[calc(100vh-4rem)] p-8 flex flex-col items-center justify-center">
          <form onSubmit={handleSubmit} className="bg-white w-[600px] rounded-lg p-8 shadow-md">
            {error && <div className="mb-4 text-red-500 text-center">{error}</div>}

            {/* Stock - Selection dynamic*/}
            <div className="mb-6">
              <label htmlFor="ticker" className="block text-lg font-light mb-2">What is the ticker?</label>
              <select
                id="ticker"
                className="w-full border border-gray-300 rounded px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
                value={formData.ticker}
                onChange={handleInputChange}
                required
              >
                <option value="">Select Ticker</option>
                <option value="TSLA">Tesla (TSLA)</option>
                <option value="F">Ford (F)</option>
                <option value="GM">General Motors (GM)</option>
                <option value="TM">Toyota (TM)</option>
                <option value="RIVN">Rivian (RIVN)</option>
              </select>
            </div>

            <div className="mb-6">
              <label htmlFor="hasStocks" className="block text-lg font-light mb-2">Do you already have stocks for this ticker?</label>
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

            {formData.hasStocks === 'yes' && (
              <div className="mb-6">
                <label htmlFor="sharesOwned" className="block text-lg font-light mb-2">How many stocks do you own?</label>
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
              <label htmlFor="cashToInvest" className="block text-lg font-light mb-2">How much cash do you want to invest in this stock? ($)</label>
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
              <button type="button" className="bg-[#DFF6D7] px-6 py-3 rounded-full text-lg font-bold shadow-md transform hover:scale-105 active:scale-95 hover:bg-green-300 active:bg-green-400 transition-all duration-200" onClick={() => navigate("/")}>RETURN</button>
              <button type="submit" className="bg-[#DFF6D7] px-6 py-3 rounded-full text-lg font-bold shadow-md transform hover:scale-105 active:scale-95 hover:bg-green-300 active:bg-green-400 transition-all duration-200 disabled:opacity-50" disabled={loading}>{loading ? 'Saving...' : 'SUBMIT'}</button>
            </div>
          </form>
        </main>
      </SidebarInset>
    </SidebarProvider>
  );
}
