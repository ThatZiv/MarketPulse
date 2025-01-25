import { createContext } from "react";

import { useState } from "react";
import { type SupabaseClient, createClient } from "@supabase/supabase-js";

const useSupabase = () => {
  const [supabase] = useState(() =>
    createClient(
      import.meta.env.VITE_SUPABASE_URL,
      import.meta.env.VITE_SUPABASE_KEY
    )
  );
  return supabase;
};

interface ISupabaseContext {
  supabase: SupabaseClient | null;
}

export const SupabaseContext = createContext<ISupabaseContext>({
  supabase: null,
});

interface SupabaseProviderProps {
  children: React.ReactNode | React.ReactNode[];
}

export const SupabaseProvider = ({ children }: SupabaseProviderProps) => {
  const supabase = useSupabase();

  return (
    <SupabaseContext.Provider value={{ supabase }}>
      {children}
    </SupabaseContext.Provider>
  );
};
