import { Route, Routes } from "react-router";
import "./App.css";
import Login from "@/pages/Login";
import Dashboard from "@/pages/Dashboard";
import Stocks from "@/pages/Stocks";
import Stock from "@/pages/Stock";
import Create from "@/pages/Create";
import { useSupabase } from "@/database/SupabaseProvider";
import { Spinner } from "@/components/ui/spinner";

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
      <Routes>
        <Route path="/" element={<Dashboard />}>
          <Route path="/stocks" element={<Stocks />} />
          <Route path="/stocks/:ticker" element={<Stock />} />
        </Route>
        <Route path="/login" element={<Login />} />
        <Route path="/create" element={<Create />} />
        <Route path="*" element={<h1>Not Found</h1>} />
      </Routes>
    </div>
  );
}

export default App;
