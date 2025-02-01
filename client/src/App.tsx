import { Route, Routes } from "react-router";
import "./App.css";
import Login from "@/pages/Login";
import Dashboard from "@/pages/Dashboard";
import Stocks from "@/pages/Stocks";
import Stock from "@/pages/Stock";
import Create from "@/pages/Create";
import { useSupabase } from "@/database/SupabaseProvider";
import { Spinner } from "@/components/ui/spinner";
import Account from "@/pages/Account";

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
      <Routes>
        <Route path="/" element={<Dashboard />}>
          <Route path="/stocks" element={<Stocks />} />
          <Route path="/stocks/:ticker" element={<Stock />} />
          <Route path="/account" element={<Account />} />
        </Route>
        <Route path="/login" element={<Login />} />
        <Route path="/create" element={<Create />} />
        <Route path="*" element={<h1>Not Found</h1>} />
      </Routes>
    </div>
  );
}

export default App;
