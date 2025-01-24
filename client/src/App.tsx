import { Route, Routes } from "react-router";
import "./App.css";
import Login from "@/pages/Login";
import Dashboard from "@/pages/Dashboard";
import Stocks from "./pages/Stocks";
import Stock from "./pages/Stock";
function App() {
  return (
    <div>
      <Routes>
        <Route path="/" element={<Dashboard />}>
          <Route path="/stocks" element={<Stocks />} />
          <Route path="/stocks/:ticker" element={<Stock />} />
        </Route>
        {/* TODO: force login to render if the user is not logged in */}
        <Route path="/login" element={<Login />} />
        <Route path="*" element={<h1>Not Found</h1>} />
      </Routes>
    </div>
  );
}

export default App;
