import React from "react";
import { SupabaseProvider } from "@/database/SupabaseProvider";

const mockSupabaseContext = {
  supabase: {
    auth: {
      signUp: jest.fn(),
      signIn: jest.fn(),
      signOut: jest.fn(),
      onAuthStateChange: jest.fn(),
    },
  },
  signUpNewUser: jest.fn(),
  signInWithEmail: jest.fn(),
  signOut: jest.fn(),
  status: "success",
  user: null,
  displayName: null,
  account: null,
  session: null,
};

export const MockSupabaseContext = React.createContext(mockSupabaseContext);

export const MockSupabaseProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => <SupabaseProvider>{children}</SupabaseProvider>;

export const useSupabase = () => mockSupabaseContext;
