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
      staleTime: Infinity,
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
                  <Toaster
                    toastOptions={{
                      unstyled: false,
                      classNames: {
                        toast: "bg-background text-foreground border-border shadow-lg",
                        error: "!text-red-600",
                      },
                    }}
                  />
                </ThemeProvider>
              </TooltipProvider>
            </ApiProvider>
          </SupabaseProvider>
        </GlobalProvider>
      </QueryClientProvider>
    </BrowserRouter>
  </StrictMode>
);
