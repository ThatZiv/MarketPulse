import { createContext } from "react";

import {
  type AuthError,
  type AuthResponse,
  type AuthTokenResponsePassword,
  type Session,
  type SupabaseClient,
  type User,
  createClient,
} from "@supabase/supabase-js";
import React from "react";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_KEY;

interface ISupabaseContext {
  supabase: SupabaseClient;
  signUpNewUser: (email: string, password: string) => Promise<AuthResponse>;
  signInWithEmail: (
    email: string,
    password: string
  ) => Promise<AuthTokenResponsePassword>;
  signOut: () => Promise<{ error: AuthError | null }>;
  isLoading: boolean;
  user: User | null;
  session: Session | null;
}

export const SupabaseContext = createContext<ISupabaseContext>({
  supabase: createClient(supabaseUrl, supabaseAnonKey),
  signUpNewUser: async () => {
    throw new Error("Supabase not initialized");
  },
  signInWithEmail: async () => {
    throw new Error("Supabase not initialized");
  },
  signOut: async () => {
    throw new Error("Supabase not initialized");
  },
  isLoading: false,
  user: null,
  session: null,
});

interface SupabaseProviderProps {
  children: React.ReactNode | React.ReactNode[];
}

export const SupabaseProvider = ({ children }: SupabaseProviderProps) => {
  const supabase = React.useMemo(
    () => createClient(supabaseUrl, supabaseAnonKey),
    []
  );

  const [isLoading, setLoading] = React.useState(true);
  const [user, setUser] = React.useState<null | User>(null);
  const [session, setSession] = React.useState<null | Session>(null);
  React.useEffect(() => {
    async function getData() {
      setLoading(true);
      try {
        // need to figure out if I need both
        const session = await supabase.auth.getSession(); // this is stored locally
        const user = await supabase.auth.getUser(); // this is a request to auth server
        setUser(user.data.user ?? null);
        setSession(session.data.session ?? null);
      } catch (error) {
        console.error("Error getting user data: ", error);
      } finally {
        setLoading(false);
      }
    }
    getData();
    const { data: authListener } = supabase.auth.onAuthStateChange(
      async (_event, session) => {
        console.log("Auth event:", _event);
        setUser(session?.user ?? null);
      }
    );

    return () => {
      authListener?.subscription.unsubscribe();
    };
  }, [supabase]);

  const signUpNewUser = React.useCallback(
    async (email: string, password: string) =>
      supabase.auth.signUp({
        email: email,
        password: password,
      }),
    [supabase]
  );

  const signInWithEmail = React.useCallback(
    async (email: string, password: string) =>
      supabase.auth.signInWithPassword({
        email: email,
        password: password,
      }),
    [supabase]
  );

  const signOut = React.useCallback(
    async () => supabase.auth.signOut(),
    [supabase]
  );

  return (
    <SupabaseContext.Provider
      value={{
        supabase,
        signUpNewUser,
        signInWithEmail,
        signOut,
        isLoading,
        user,
        session,
      }}
    >
      {children}
    </SupabaseContext.Provider>
  );
};

export const useSupabase = () =>
  React.useContext<ISupabaseContext>(SupabaseContext);
