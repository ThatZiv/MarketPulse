import { ResetPasswordForm } from "@/components/reset_password";
import { describe } from "node:test";
import {
  act,
  render,
  screen,
  cleanup,
  fireEvent,
} from "@testing-library/react";
import "@testing-library/jest-dom";
import { BrowserRouter } from "react-router-dom";
import { Toaster } from "sonner";

afterEach(() => {
  cleanup();
});
const mockSupabase = jest.fn(() => {
  return { error: null };
});

const mockToast = jest.fn();

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
          updateUser: mockSupabase,
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

describe("Forgot Password", () => {
  test("Forgot Password rendering", async () => {
    render(
      <BrowserRouter>
        <ResetPasswordForm />
      </BrowserRouter>
    );

    const title = await screen.findByText("Reset Password");
    const reset = await screen.findByText("Reset");
    const back = await screen.findByText("Back");
    const password = await screen.findByPlaceholderText("Password");
    const confirmPassword =
      await screen.findByPlaceholderText("Confirm Password");

    expect(title).toBeInTheDocument();
    expect(reset).toBeInTheDocument();
    expect(back).toBeInTheDocument();
    expect(password).toBeInTheDocument();
    expect(confirmPassword).toBeInTheDocument();
  });

  test("Back button", async () => {
    render(
      <BrowserRouter>
        <ResetPasswordForm />
      </BrowserRouter>
    );
    const back = await screen.findByText("Back");
    expect(back).toBeInTheDocument();
    expect(back).not.toBeDisabled();
    expect(back).toBeVisible();

    await act(async () => {
      await fireEvent.click(back);
    });

    expect(mockSignOut).toHaveBeenCalled();
  });

  test("Reset button", async () => {
    jest.mock("sonner", () => ({
      toast: () => mockToast,
    }));

    render(
      <BrowserRouter>
        <ResetPasswordForm />
        <Toaster />
      </BrowserRouter>
    );
    const reset = await screen.findByText("Reset");
    const password = await screen.findByPlaceholderText("Password");
    const confirmPassword =
      await screen.findByPlaceholderText("Confirm Password");
    expect(reset).toBeInTheDocument();
    expect(reset).not.toBeDisabled();
    expect(reset).toBeVisible();
    expect(password).toBeInTheDocument();
    expect(password).not.toBeDisabled();
    expect(password).toBeVisible();
    expect(confirmPassword).toBeInTheDocument();
    expect(confirmPassword).not.toBeDisabled();
    expect(confirmPassword).toBeVisible();

    await act(async () => {
      await fireEvent.change(password, { target: { value: "Password123!" } });
    });

    await act(async () => {
      await fireEvent.change(confirmPassword, {
        target: { value: "Password123!" },
      });
    });

    await act(async () => {
      await fireEvent.click(reset);
    });

    expect(
      await screen.findByText("✅ At least 8 characters")
    ).toBeInTheDocument();
    expect(
      await screen.findByText("✅ At least 1 uppercase letter")
    ).toBeInTheDocument();
    expect(await screen.findByText("✅ At least 1 number")).toBeInTheDocument();
    expect(
      await screen.findByText("✅ At least 1 special character")
    ).toBeInTheDocument();
  });
});
