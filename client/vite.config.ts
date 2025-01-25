import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { z } from "zod";
import { ValidateEnv } from "@julr/vite-plugin-validate-env";

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    ValidateEnv({
      validator: "zod",
      schema: {
        VITE_API_URL: z.string().url(),
        VITE_SUPABASE_URL: z.string().url(),
        VITE_SUPABASE_KEY: z.string().nonempty(),
        VITE_GOOGLE_CLIENT_ID: z.string().nonempty(),
      },
    }),
  ],
  server: {
    allowedHosts: process.env.NODE_ENV === "development" ? true : undefined,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
