import { ResetPassword } from "@/components/forgot-password";

import UserAuth from "@/pages/UserAuth";
import { BrowserRouter } from "react-router-dom";
import { describe } from "node:test";
import {
  act,
  render,
  screen,
  cleanup,
  fireEvent,
} from "@testing-library/react";

import "@testing-library/jest-dom";

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

jest.mock("lucide-react", () => ({
  Moon: () => "MoonIcon",
  Sun: () => "SunIcon",
  Eye: () => "EyeIcon",
  EyeOff: () => "EyeOffIcon",
}));

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
describe("Forgot Password", () => {
  const login = jest.fn();

  test("Forgot Password rendering", async () => {
    render(<ResetPassword resetPasswordState={login} />);
    const title = await screen.findByTestId("title");
    const reset = await screen.findByText("Recover");
    const back = await screen.findByText("Back");
    const email = await screen.findByPlaceholderText("Email Address");

    expect(title).toBeInTheDocument();
    expect(reset).toBeInTheDocument();
    expect(back).toBeInTheDocument();
    expect(email).toBeInTheDocument();
  });

  test("Back button", async () => {
    render(<ResetPassword resetPasswordState={login} />);
    const back = await screen.findByText("Back");
    expect(back).toBeInTheDocument();
    expect(back).not.toBeDisabled();
    expect(back).toBeVisible();

    await act(async () => {
      await fireEvent.click(back);
    });

    expect(login).toHaveBeenCalledTimes(1);
  });

  test("Reset button", async () => {
    render(<ResetPassword resetPasswordState={login} />);
    const reset = await screen.findByText("Recover");
    expect(reset).toBeInTheDocument();
    expect(reset).not.toBeDisabled();
    expect(reset).toBeVisible();

    const email = await screen.findByPlaceholderText("Email Address");
    await act(async () => {
      await fireEvent.change(email, { target: { value: "bob@bob.com" } });
    });

    await act(async () => {
      await fireEvent.click(reset);
    });
    expect(mockSupabase).toHaveBeenCalledTimes(1);
    expect(login).toHaveBeenCalledTimes(2);
  });

  test("Reset Button Failure", async () => {
    render(<ResetPassword resetPasswordState={login} />);
    const reset = await screen.findByText("Recover");
    expect(reset).toBeInTheDocument();
    expect(reset).not.toBeDisabled();
    expect(reset).toBeVisible();

    await act(async () => {
      await fireEvent.click(reset);
    });
    expect(mockSupabase).toHaveBeenCalledTimes(1);
    expect(login).toHaveBeenCalledTimes(2);
  });

  test("Bad Email", async () => {
    render(<ResetPassword resetPasswordState={login} />);
    const reset = await screen.findByText("Recover");
    expect(reset).toBeInTheDocument();
    expect(reset).not.toBeDisabled();
    expect(reset).toBeVisible();

    const email = await screen.findByPlaceholderText("Email Address");
    await act(async () => {
      await fireEvent.change(email, { target: { value: "bob@bob.com" } });
    });

    await act(async () => {
      await fireEvent.click(reset);
    });
  });

  test("Access and return exit Forgot Password page", async () => {
    render(
      <BrowserRouter>
        <UserAuth />
      </BrowserRouter>
    );

    const button = await screen.findByText("Forgot Password?");
    expect(button).toBeInTheDocument();

    await act(async () => {
      await fireEvent.click(button);
    });

    const back = await screen.findByText("Back");
    const title = await screen.findByText("Recover Password");

    expect(back).toBeInTheDocument();
    expect(title).toBeInTheDocument();

    await act(async () => {
      await fireEvent.click(back);
    });

    const log_in = await screen.findByText("Log in");

    expect(log_in).toBeInTheDocument();
  });
});
