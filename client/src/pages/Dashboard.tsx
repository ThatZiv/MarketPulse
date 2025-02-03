import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Settings } from "lucide-react";
import { useSupabase } from "@/database/SupabaseProvider";
import { 
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { Separator } from "@/components/ui/separator";

import { Outlet, useLocation } from "react-router";
import { useMemo } from "react";

export default function Dashboard() {
  const navigate = useNavigate();
  const { user } = useSupabase();
  
  // Handle navigation to the login page when settings button is clicked
  const handleSettingsClick = () => {
    navigate("/login");
  };


  return (
    <SidebarProvider>
      {/* Sidebar Navigation */}
      <AppSidebar />
      <SidebarInset>
        {/* Header Component */}
        <Header 
          userName={user?.email ?? "Guest"}
          onSettingsClick={handleSettingsClick}
        />
        {/* Main Dashboard Content */}
        <MainContent />
      </SidebarInset>
    </SidebarProvider>
  );
}

// Header component displaying user information and settings button
const Header: React.FC<HeaderProps> = ({ userName, onSettingsClick }) =>  {
  return (
    <header className="relative h-16 bg-[#F5F5F5] px-4 border-b border-gray-200 flex items-center">
      <div className="flex items-center gap-4">
        <SidebarTrigger className="h-6 w-6 text-black" />
        <Separator orientation="vertical" className="h-4" />
      </div>
      <h1 className="text-4xl font-[Poppins] font-bold text-center flex-1 tracking-tight text-gray-800">
        {userName || 'User'} Dashboard
      </h1>
      <div className="absolute right-4 h-full flex items-center">
        <button
          className="flex items-center justify-center p-2 rounded-full 
                   transform hover:scale-105 active:scale-95 
                   hover:bg-gray-100 active:bg-gray-200 
                   transition-all duration-200"
          onClick={onSettingsClick}
        >
          <Settings className="h-6 w-6 text-gray-800" />
        </button>
      </div>
    </header>
  );
}

// Main dashboard content, fetching and displaying user stocks
const MainContent: React.FC = () => {
  const { user, supabase } = useSupabase();
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch user's stock investments from the database
    const fetchUserStocks = async () => {
      try {
        const { data, error } = await supabase
          .from('User_Stocks')
          .select('stock_id, shares_owned')
          .eq('user_id', user?.id)
          .order('created_at', { ascending: false })
          .limit(5);

        if (error) throw error;
        setStocks(data || []);
      } catch (error) {
        console.error('Error fetching user stocks:', error);
      } finally {
        setLoading(false);
      }
    };

    if (user) fetchUserStocks();
  }, [user, supabase]);

  return (
    <main className="bg-[#F5F5F5] min-h-[calc(100vh-4rem)] p-8 flex flex-col">
      <div className="flex flex-col items-center gap-8 flex-grow">
        {/* Button to add new stocks */}
        <Link
          className="flex items-center justify-center h-16 w-16 rounded-full bg-[#DFF6D7] text-4xl font-bold shadow hover:shadow-md transition-transform transform hover:scale-105 active:scale-95"
          to="/stocks"
        >
          +
        </Link>

        {/* Display user's investment portfolio */}
        <section className="w-full">
          <h2 className="text-2xl font-light mb-6 text-center">
            Your Investment Portfolio:
          </h2>
          
          {loading ? (
            <div className="text-center">Loading investments...</div>
          ) : stocks.length === 0 ? (
            <div className="text-center">No investments found, click the "+" to add your first investment</div>
          ) : (
            <div className="flex flex-col items-center gap-6">
              <div className="grid grid-cols-2 gap-6 w-full max-w-4xl">
                {stocks.slice(0, 2).map((stock) => (
                  <StockCard key={stock.stock_id} stock={stock} />
                ))}
              </div>
              <div className="grid grid-cols-3 gap-6 w-full max-w-4xl">
                {stocks.slice(2, 5).map((stock) => (
                  <StockCard key={stock.stock_id} stock={stock} />
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

// Stock data structure
interface Stock {
  stock_id: string;
  shares_owned: number;
}

// Props for StockCard component
interface StockCardProps {
  stock: Stock;
}

// Component for displaying individual stock information
function StockCard({ stock }: StockCardProps) {
  return (
    <div className="bg-[#DFF6D7] p-6 rounded-lg shadow flex flex-col justify-center items-center text-center hover:shadow-md transition-shadow">
      <h3 className="text-lg font-bold uppercase tracking-wide mb-4">
        {stock.stock_id}
      </h3>
      <p className="text-sm font-medium">
        Shares Owned: {stock.shares_owned.toLocaleString()}
      </p>
    </div>
  );
}

// Props for Header component
interface HeaderProps {
  userName: string;
  onSettingsClick: () => void;
}
