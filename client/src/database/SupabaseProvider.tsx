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
import { useNavigate, useLocation } from "react-router";
import { toast } from "sonner";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_KEY;
const supabaseClient = createClient(supabaseUrl, supabaseAnonKey);

type googleResponse = {
  clientId: string;
  client_id: string;
  credential: string;
  select_by: string;
};

const nonAuthenticatedRoutes = ["/create", "/login"]

interface ISupabaseContext {
  supabase: SupabaseClient;
  signUpNewUser: (email: string, password: string) => Promise<AuthResponse>;
  signInWithEmail: (
    email: string,
    password: string
  ) => Promise<AuthTokenResponsePassword>;
  signInWithGoogle: (
    data: googleResponse
  ) => Promise<AuthTokenResponsePassword>;
  signOut: () => Promise<{ error: AuthError | null }>;
  isLoading: boolean;
  user: User | null;
  session: Session | null;
}

export const SupabaseContext = createContext<ISupabaseContext>({
  supabase: supabaseClient,
  signUpNewUser: async () => {
    throw new Error("Supabase not initialized");
  },
  signInWithEmail: async () => {
    throw new Error("Supabase not initialized");
  },
  signInWithGoogle: async () => {
    throw new Error("Supabase not initialized");
  },
  signOut: async () => {
    throw new Error("Supabase not initialized");
  },
  isLoading: true,
  user: null,
  session: null,
});

interface SupabaseProviderProps {
  children: React.ReactNode | React.ReactNode[];
}

export const SupabaseProvider = ({ children }: SupabaseProviderProps) => {
  const supabase = React.useMemo(() => supabaseClient, []);

  const navigate = useNavigate();
  const location = useLocation()
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
        setUser(session?.user ?? null);
      }
    );

    return () => {
      authListener?.subscription.unsubscribe();
    };
  }, [supabase]);

  React.useEffect(() => {
    // TODO: there prob is a better way to do this (middleware/auth comps)
    if (!isLoading && !session && !nonAuthenticatedRoutes.includes(location.pathname)) {
      navigate("/login");
    }
  }, [session, isLoading, navigate, location]);

  const signUpNewUser = React.useCallback(
    async (email: string, password: string) => {
      const signUp = new Promise<AuthResponse>((resolve, reject) => {
        supabase.auth
          .signUp({
            email: email,
            password: password,
          })
          .then(({ data, error }) => {
            return error ? reject(error) : resolve({ data, error });
          })
          .catch((error) => {
            return reject(error);
          });
      });
      toast.promise(signUp, {
        loading: "Signing up...",
        success: (data) => {
          setUser(data.data.user);
          setSession(data.data.session);
          navigate("/login");
          return "Signed up! Please check your email to verify your account.";
        },
        error: (error) => {
          console.error("Error signing up: ", error);
          return "Error signing up: " + error.message;
        },
      });
      return signUp;
    },
    [navigate, supabase.auth]
  );

  const signInWithEmail = React.useCallback(
    async (email: string, password: string) => {
      const signIn = new Promise<AuthTokenResponsePassword>(
        (resolve, reject) => {
          supabase.auth
            .signInWithPassword({
              email: email,
              password: password,
            })
            .then(({ data, error }) => {
              return error ? reject(error) : resolve({ data, error });
            })
            .catch((error) => {
              return reject(error);
            });
        }
      );

      toast.promise(signIn, {
        loading: "Logging in...",
        success: (data) => {
          setUser(data.data.user);
          setSession(data.data.session);
          navigate("/");
          return "Logged in!";
        },
        error: (error) => {
          console.error("Error logging in: ", error);
          return "Error logging in: " + error.message;
        },
      });

      return signIn;
    },
    [navigate, supabase.auth]
  );

  const signInWithGoogle = React.useCallback(
    async (data: googleResponse) => {
      const signIn = new Promise<AuthTokenResponsePassword>(
        (resolve, reject) => {
          supabase.auth
            .signInWithIdToken({
              provider: "google",
              token: data.credential,
            })
            .then(({ data, error }) => {
              return error ? reject(error) : resolve({ data, error });
            })
            .catch((error) => {
              return reject(error);
            });
        }
      );

      toast.promise(signIn, {
        loading: "Logging in...",
        success: (data) => {
          setUser(data.data.user);
          setSession(data.data.session);
          navigate("/");
          return "Logged in!";
        },
        error: (error) => {
          console.error("Error logging in: ", error);
          return "Error logging in: " + error.message;
        },
      });

      return signIn;
    },
    [navigate, supabase.auth]
  );

  const signOut = React.useCallback(async () => {
    const res = await supabase.auth.signOut();
    setUser(null);
    setSession(null);
    navigate("/login");
    return res;
  }, [navigate, supabase.auth]);

  return (
    <SupabaseContext.Provider
      value={{
        supabase,
        signUpNewUser,
        signInWithEmail,
        signInWithGoogle,
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
