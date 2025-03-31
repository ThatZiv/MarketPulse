import { Route, Routes } from "react-router";
import "./App.css";
import UserAuth from "@/pages/UserAuth";
import Dashboard from "@/pages/Dashboard";
import StockSelection from "@/pages/StockSelection";
import StockEstimation from "@/pages/StockEstimation";
import { useSupabase } from "@/database/SupabaseProvider";
import { Spinner } from "@/components/ui/spinner";
import Settings from "@/pages/Settings";
import Landing from "@/pages/Landing";
import Introduction from "@/pages/Introduction";
import Documentation from "@/pages/Documentation";
import Tutorials from "@/pages/Tutorials";
import FAQ from "@/pages/FAQ";
import Disclaimer from "./pages/Disclaimers";
import Feedback from "@/pages/Feedback";
import Support from "@/pages/Support";



import ResetPasswordPage from "./pages/ResetPasswordPage";
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
        <Route path="/reset" element={<ResetPasswordPage />} />
        <Route path="/" element={<Dashboard />}>
          <Route index element={<Landing />} />
          <Route path="/stocks" element={<StockSelection />} />
          <Route path="/stocks/:ticker" element={<StockEstimation />} />
          <Route path="/feedback" element={<Feedback />} />
          <Route path="/support" element={<Support />} />

          <Route path="/settings">
            <Route path=":tab" element={<Settings />} />
            <Route index element={<Settings />} />
          </Route>
          <Route path="/documentation" element={<Documentation />} />
          <Route
            path="/documentation/introduction"
            element={<Introduction />}
          />
          <Route path="/documentation/faq" element={<FAQ />} />
          <Route path="/documentation/tutorials" element={<Tutorials />} />
          <Route path="/documentation/disclaimer" element={<Disclaimer />} />
        </Route>
        <Route path="/auth" element={<UserAuth />} />
        <Route path="*" element={<h1>Not Found</h1>} />
      </Routes>
    </div>
  );
}

export default App;
