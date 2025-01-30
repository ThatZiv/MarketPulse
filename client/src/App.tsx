import { Route, Routes } from "react-router";
import "./App.css";
import UserAuth from "@/pages/UserAuth";
import Dashboard from "@/pages/Dashboard";
import StockSelection from "@/pages/StockSelection";
import StockEstimation from "@/pages/StockEstimation";
import { useSupabase } from "@/database/SupabaseProvider";
import { Spinner } from "@/components/ui/spinner";
import { ThemeProvider } from "@/components/ui/theme-provider"

function App() {
  const { isLoading } = useSupabase();
  if (isLoading)
    return (
      <div className="flex justify-center items-center h-screen">
        <Spinner />
      </div>
    );
  return (
    <div>
      <ThemeProvider defaultTheme="system" storageKey="vite-ui-theme">

      <Routes>
        <Route path="/" element={<Dashboard />}>
          <Route path="/stocks" element={<StockSelection />} />
          <Route path="/stocks/:ticker" element={<StockEstimation />} />
        </Route>
        <Route path="/userAuth" element={<UserAuth />} />
        <Route path="*" element={<h1>Not Found</h1>} />
      </Routes>
      </ThemeProvider>

    </div>
  );
}

export default App;
