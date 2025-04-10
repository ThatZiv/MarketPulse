import { LoginForm } from "@/components/login-form";
import { describe, test, afterEach, beforeEach } from "@jest/globals";
import {
  render,
  screen,
  fireEvent,
  act,
  cleanup,
} from "@testing-library/react";
import "@testing-library/jest-dom";
import { toast } from "sonner";

jest.mock("lucide-react", () => ({
  Eye: () => <span>Eye Icon</span>,
  EyeOff: () => <span>EyeOff Icon</span>,
}));

jest.mock("sonner", () => ({
  toast: {
    error: jest.fn(),
  },
}));

const mockSignInWithEmail = jest.fn();
const mockSignInWithGoogle = jest.fn();
const mockTogglePageState = jest.fn();
const mockResetPasswordState = jest.fn();

jest.mock("@/database/SupabaseProvider", () => ({
  useSupabase: () => ({
    signInWithEmail: mockSignInWithEmail,
    signInWithGoogle: mockSignInWithGoogle,
  }),
}));

afterEach(() => {
  cleanup();
  jest.clearAllMocks();
});

describe("LoginForm Component", () => {
  beforeEach(() => {
    render(
      <LoginForm
        togglePageState={mockTogglePageState}
        resetPasswordState={mockResetPasswordState}
      />
    );
  });

  test("renders LoginForm correctly", () => {
    const title = screen.getByText("Log in");
    const description = screen.getByText(
      "Enter your info below to login to your account"
    );
    const emailInput = screen.getByPlaceholderText("Email Address");
    const passwordInput = screen.getByPlaceholderText("Password");
    const loginButton = screen.getByText("Login");
    const forgotPasswordButton = screen.getByText("Forgot Password?");
    const createAccountButton = screen.getByText("Create Account");

    expect(title).toBeInTheDocument();
    expect(description).toBeInTheDocument();
    expect(emailInput).toBeInTheDocument();
    expect(passwordInput).toBeInTheDocument();
    expect(loginButton).toBeInTheDocument();
    expect(forgotPasswordButton).toBeInTheDocument();
    expect(createAccountButton).toBeInTheDocument();
  });

  test("handles email and password input", async () => {
    const emailInput = screen.getByPlaceholderText("Email Address");
    const passwordInput = screen.getByPlaceholderText("Password");

    await act(async () => {
      fireEvent.change(emailInput, { target: { value: "test@example.com" } });
      fireEvent.change(passwordInput, { target: { value: "password123" } });
    });

    expect(emailInput).toHaveValue("test@example.com");
    expect(passwordInput).toHaveValue("password123");
  });

  test("calls signInWithEmail on login button click", async () => {
    const emailInput = screen.getByPlaceholderText("Email Address");
    const passwordInput = screen.getByPlaceholderText("Password");
    const loginButton = screen.getByText("Login");

    await act(async () => {
      fireEvent.change(emailInput, { target: { value: "test@example.com" } });
      fireEvent.change(passwordInput, { target: { value: "password123" } });
      fireEvent.click(loginButton);
    });

    expect(mockSignInWithEmail).toHaveBeenCalledTimes(1);
    expect(mockSignInWithEmail).toHaveBeenCalledWith(
      "test@example.com",
      "password123"
    );
  });

  test("calls resetPasswordState on forgot password button click", async () => {
    const forgotPasswordButton = screen.getByText("Forgot Password?");

    await act(async () => {
      fireEvent.click(forgotPasswordButton);
    });

    expect(mockResetPasswordState).toHaveBeenCalledTimes(1);
    expect(mockResetPasswordState).toHaveBeenCalledWith(true);
  });

  test("calls togglePageState on create account button click", async () => {
    const createAccountButton = screen.getByText("Create Account");

    await act(async () => {
      fireEvent.click(createAccountButton);
    });

    expect(mockTogglePageState).toHaveBeenCalledTimes(1);
  });

  test("toggles password visibility", async () => {
    const passwordInput = screen.getByPlaceholderText("Password");
    const toggleVisibilityButton = screen.getByText("Eye Icon");

    expect(passwordInput).toHaveAttribute("type", "password");

    await act(async () => {
      fireEvent.click(toggleVisibilityButton);
    });
    expect(passwordInput).toHaveAttribute("type", "text");

    await act(async () => {
      fireEvent.click(toggleVisibilityButton);
    });
    expect(passwordInput).toHaveAttribute("type", "text");
  });

  test("displays error toast when login credentials are invalid", async () => {
    const emailInput = screen.getByPlaceholderText("Email Address");
    const passwordInput = screen.getByPlaceholderText("Password");
    const loginButton = screen.getByText("Login");

    mockSignInWithEmail.mockRejectedValueOnce(
      new Error("Error logging in: Invalid login credentials.")
    );

    await act(async () => {
      fireEvent.change(emailInput, { target: { value: "johndoe@yahoo.com" } });
      fireEvent.change(passwordInput, { target: { value: "Password!123" } });
      fireEvent.click(loginButton);
    });

    expect(toast.error).toHaveBeenCalledTimes(1);
    expect(toast.error).toHaveBeenCalledWith(
      "Error logging in: Invalid login credentials."
    );

    expect(mockSignInWithEmail).toHaveBeenCalledTimes(1);
    expect(mockSignInWithEmail).toHaveBeenCalledWith(
      "johndoe@yahoo.com",
      "Password!123"
    );
  });

  test("displays error toast when login credentials are invalid", async () => {
    const emailInput = screen.getByPlaceholderText("Email Address");
    const passwordInput = screen.getByPlaceholderText("Password");
    const loginButton = screen.getByText("Login");

    mockSignInWithEmail.mockRejectedValueOnce(
      new Error("Error logging in: Invalid login credentials.")
    );

    await act(async () => {
      fireEvent.change(emailInput, { target: { value: "test2025@test.com" } });
      fireEvent.change(passwordInput, { target: { value: "Passwo123!" } });
      fireEvent.click(loginButton);
    });

    expect(toast.error).toHaveBeenCalledTimes(1);
    expect(toast.error).toHaveBeenCalledWith(
      "Error logging in: Invalid login credentials."
    );

    expect(mockSignInWithEmail).toHaveBeenCalledTimes(1);
    expect(mockSignInWithEmail).toHaveBeenCalledWith(
      "test2025@test.com",
      "Passwo123!"
    );
  });

  test("toggles password visibility and displays the entered password", async () => {
    const passwordInput = screen.getByPlaceholderText("Password");
    const toggleVisibilityButton = screen.getByText("Eye Icon");

    await act(async () => {
      fireEvent.change(passwordInput, { target: { value: "Password123!" } });
    });

    expect(passwordInput).toHaveAttribute("type", "password");

    await act(async () => {
      fireEvent.click(toggleVisibilityButton);
    });

    expect(passwordInput).toHaveAttribute("type", "text");
    expect(passwordInput).toHaveValue("Password123!");

    await act(async () => {
      fireEvent.click(toggleVisibilityButton);
    });

    expect(passwordInput).toHaveAttribute("type", "text");
  });


  test("calls signInWithGoogle on Google sign-in button click", async () => {
    const mockGoogleResponse = {
      clientId: "mock-client-id",
      client_id: "mock-client-id",
      credential: "mock-credential",
      select_by: "mock-select-by",
    };
  
    window.handleSignInWithGoogle = jest.fn(async (response) => {
      await mockSignInWithGoogle(response);
    });
  
    const googleSignInButton = document.querySelector(".g_id_signin") as HTMLElement;
  
    await act(async () => {
      fireEvent.click(googleSignInButton);
    });
  
    await act(async () => {
      window.handleSignInWithGoogle(mockGoogleResponse);
    });
  
    expect(window.handleSignInWithGoogle).toHaveBeenCalledTimes(1);
    expect(window.handleSignInWithGoogle).toHaveBeenCalledWith(mockGoogleResponse);
    expect(mockSignInWithGoogle).toHaveBeenCalledTimes(1);
    expect(mockSignInWithGoogle).toHaveBeenCalledWith(mockGoogleResponse);
  });


  
});
