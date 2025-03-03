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
import { useGlobal } from "@/lib/GlobalProvider";
import { actions } from "@/lib/constants";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_KEY;
const supabaseClient = createClient(supabaseUrl, supabaseAnonKey);

type googleResponse = {
  clientId: string;
  client_id: string;
  credential: string;
  select_by: string;
};

type Status = "loading" | "error" | "success";
const nonAuthenticatedRoutes = ["auth"];

interface IAccount {
  first_name?: string;
  last_name?: string;
}
export interface ISupabaseContext {
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
  status: Status;
  user: User | null;
  displayName: string;
  account: IAccount | null;
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
  status: "loading",
  account: null,
  displayName: "User",
  user: null,
  session: null,
});

interface SupabaseProviderProps {
  children: React.ReactNode | React.ReactNode[];
}

export const SupabaseProvider = ({ children }: SupabaseProviderProps) => {
  const supabase = React.useMemo(() => supabaseClient, []);

  const navigate = useNavigate();
  const location = useLocation();
  const { state, dispatch } = useGlobal();
  const [status, setStatus] = React.useState<Status>("loading");
  const [account, setAccount] = React.useState<null | IAccount>(null);
  const [user, setUser] = React.useState<null | User>(null);
  const [session, setSession] = React.useState<null | Session>(null);
  React.useEffect(() => {
    async function getData() {
      setStatus("loading");
      try {
        // need to figure out if I need both
        const session = await supabase.auth.getSession(); // this is stored locally
        const user = await supabase.auth.getUser(); // this is a request to auth server
        const thisUser = user.data.user ?? null;
        setUser(thisUser ?? null);
        setSession(session.data.session ?? null);
        if (thisUser) {
          // preload account "profile" data
          const { data: accountData } = await supabase
            .from("Account")
            .select(`first_name, last_name`)
            .eq("user_id", thisUser.id)
            .maybeSingle();
          setAccount(accountData ?? null);
          dispatch({
            type: actions.SET_USER_FULL_NAME,
            payload: accountData
              ? [accountData.first_name, accountData.last_name]
              : null,
          });
        }
        setStatus("success");
      } catch (error) {
        console.error("Error getting user data: ", error);
        setStatus("error");
      }
    }
    getData();
    const { data: authListener } = supabase.auth.onAuthStateChange(
      async (_event, session) => {
        setUser(session?.user ?? null);
        setSession(session ?? null);
      }
    );

    return () => {
      authListener?.subscription.unsubscribe();
    };
  }, [supabase]);

  const displayName = React.useMemo(() => {
    if (!state.user.name) return user?.email ?? "User";
    // FIXME: this doesnt change
    return state.user.name;
  }, [state.user, user]);
  //   return [account.first_name, account.last_name]
  //     .filter((x) => x)
  //     .join(" ")
  //     .trim();
  // }, [account, user]);

  React.useEffect(() => {
    // TODO: there prob is a better way to do this (middleware/auth comps)
    if (
      status !== "loading" &&
      !session &&
      !nonAuthenticatedRoutes.includes(location.pathname)
    ) {
      navigate("/auth");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session, status]);

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
          navigate("/auth");
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
    navigate("/auth");
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
        status,
        account,
        displayName,
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
