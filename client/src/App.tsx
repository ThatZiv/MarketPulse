import { Route, Routes } from "react-router";
import "./App.css";
import UserAuth from "@/pages/UserAuth";
import Dashboard from "@/pages/Dashboard";
import StockSelection from "@/pages/StockSelection";
import StockEstimation from "@/pages/StockEstimation";
import { useSupabase } from "@/database/SupabaseProvider";
import { Spinner } from "@/components/ui/spinner";
import Account from "@/pages/Account";
import { ThemeProvider } from "@/components/ui/theme-provider";

function App() {
  const { status } = useSupabase();
  if (status == "loading")
    return (
      <div className="flex justify-center items-center h-screen">
        <Spinner />
      </div>
    );
  else if (status == "error")
    return (
      <div className="flex flex-col justify-center items-center h-screen">
        <h1 className="text-3xl">Error</h1>
        <p className="text-gray-600">
          Unfortunately, we encountered an error. Please refresh the page or try
          again later.
        </p>
      </div>
    );
  return (
    <div>
      <ThemeProvider defaultTheme="system" storageKey="vite-ui-theme">
        <Routes>
          <Route path="/" element={<Dashboard />}>
            <Route path="/stocks" element={<StockSelection />} />
            <Route path="/stocks/:ticker" element={<StockEstimation />} />
            <Route path="/account" element={<Account />} />
          </Route>
          <Route path="/auth" element={<UserAuth />} />
          <Route path="*" element={<h1>Not Found</h1>} />
        </Routes>
      </ThemeProvider>
    </div>
  );
}

export default App;
