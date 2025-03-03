import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { BrowserRouter } from "react-router";
import { SupabaseProvider } from "./database/SupabaseProvider.tsx";
import { Toaster } from "@/components/ui/sonner.tsx";
import { ThemeProvider } from "@/components/ui/theme-provider";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiProvider } from "@/lib/ApiProvider.tsx";
import { TooltipProvider } from "@/components/ui/tooltip.tsx";
import { GlobalProvider } from "@/lib/GlobalProvider.tsx";

const queryClient = new QueryClient();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>
        <SupabaseProvider>
          <ApiProvider>
            <GlobalProvider>
              <TooltipProvider>
                <ThemeProvider defaultTheme="system" storageKey="vite-ui-theme">
                  <App />
                  <Toaster />
                </ThemeProvider>
              </TooltipProvider>
            </GlobalProvider>
          </ApiProvider>
        </SupabaseProvider>
      </QueryClientProvider>
    </BrowserRouter>
  </StrictMode>
);
