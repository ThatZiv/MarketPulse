import { ResetPassword } from "@/components/forgot-password";
import { describe } from "node:test";
import {
  act,
  render,
  screen,
  cleanup,
  fireEvent,
} from "@testing-library/react";

import "@testing-library/jest-dom";

import { GenerateStockLLM } from "@/components/llm/stock-llm";

afterEach(() => {
  cleanup();
});
const mockSupabase = jest.fn(() => {
  return { error: null };
});
const mockSignUpNewUser = jest.fn();
const mockSignInWithEmail = jest.fn();
const mockSignInWithGoogle = jest.fn();
const mockSignOut = jest.fn();
const mockStatus = jest.fn();
const mockUser = jest.fn();
const mockName = jest.fn();
const mockAccount = jest.fn();
const mockSession = jest.fn();
jest.mock("@/database/SupabaseProvider", () => ({
  useSupabase: () => {
    return {
      supabase: {
        auth: {
          resetPasswordForEmail: mockSupabase,
        },
      },
      signUpNewUser: mockSignUpNewUser,
      signInWithEmail: mockSignInWithEmail,
      signInWithGoogle: mockSignInWithGoogle,
      signOut: mockSignOut,
      status: mockStatus,
      user: mockUser,
      displayName: mockName,
      account: mockAccount,
      session: mockSession,
    };
  },
}));
jest.mock("lucide-react", () => ({
    Moon: () => "MoonIcon",
    Sun: () => "SunIcon",
    Eye: () => "EyeIcon",
    EyeOff: () => "EyeOffIcon",
  }));

jest.mock("@/lib/ApiProvider", () => ({
    useApi: () => jest.fn()
}))

describe("MarketPulse AI", () => {
    test("Render", async () => {
        const ticker = "test"
        render(<GenerateStockLLM ticker={ticker} />);
    
      });
});