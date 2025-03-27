import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { BrowserRouter } from "react-router";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { SupabaseProvider } from "./database/SupabaseProvider.tsx";
import { Toaster } from "@/components/ui/sonner.tsx";
import { ThemeProvider } from "@/components/ui/theme-provider";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiProvider } from "@/lib/ApiProvider.tsx";
import { TooltipProvider } from "@/components/ui/tooltip.tsx";
import { GlobalProvider } from "@/lib/GlobalProvider.tsx";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // TODO: figure out why invalidate cache not triggering refetch
      staleTime: 0,
      gcTime: Infinity,
      refetchOnWindowFocus: false,
      refetchOnMount: false,
      refetchOnReconnect: true,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>
        {import.meta.env.DEV && <ReactQueryDevtools initialIsOpen={false} />}
        <GlobalProvider>
          <SupabaseProvider>
            <ApiProvider>
              <TooltipProvider delayDuration={100}>
                <ThemeProvider defaultTheme="system" storageKey="vite-ui-theme">
                  <App />
                  <Toaster />
                </ThemeProvider>
              </TooltipProvider>
            </ApiProvider>
          </SupabaseProvider>
        </GlobalProvider>
      </QueryClientProvider>
    </BrowserRouter>
  </StrictMode>
);
