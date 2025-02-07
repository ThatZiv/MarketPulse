import React from "react";
import { useSupabase } from "@/database/SupabaseProvider";
import Api, { type IApi } from "./api";

export const useApi = () => React.useContext<IApi | null>(ApiContext);

export const ApiContext = React.createContext<IApi | null>(null);

export const ApiProvider = ({ children }: { children: React.ReactNode }) => {
  const { session } = useSupabase();
  const api = React.useMemo(() => new Api(session!.access_token), [session]);

  return <ApiContext.Provider value={api}>{children}</ApiContext.Provider>;
};
