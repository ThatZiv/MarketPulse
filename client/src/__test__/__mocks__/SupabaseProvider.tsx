import React from "react";
import { SupabaseContext } from "@/database/SupabaseProvider";

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
  accoubt: null,
  session: null,
};

export const MockSupabaseProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => (
  <SupabaseContext.Provider value={mockSupabaseContext as never}>
    {children}
  </SupabaseContext.Provider>
);

export const useSupabase = () => mockSupabaseContext;
