import { CreateForm } from "@/components/create_user_form";
import { describe, test, afterEach, beforeEach } from "@jest/globals";
import {
  render,
  screen,
  fireEvent,
  act,
  cleanup,
} from "@testing-library/react";
import "@testing-library/jest-dom";

jest.mock("lucide-react", () => ({
    Eye: () => <span>Eye Icon</span>,
    EyeOff: () => <span>EyeOff Icon</span>,
  }));

const mockSignUpNewUser = jest.fn();
const mockSignInWithGoogle = jest.fn();
const mockTogglePageState = jest.fn();

jest.mock("@/database/SupabaseProvider", () => ({
  useSupabase: () => ({
    signUpNewUser: mockSignUpNewUser,
    signInWithGoogle: mockSignInWithGoogle,
  }),
}));

afterEach(() => {
  cleanup();
  jest.clearAllMocks();
});

describe("CreateForm Component", () => {
  beforeEach(() => {
    document.body.innerHTML = `
      <div id="g_id_onload"></div>
      <div class="g_id_signin"></div>
    `;
    render(<CreateForm togglePageState={mockTogglePageState} />);
  });

  test("renders CreateForm correctly", () => {
    const title = screen.getByText("Create an Account");
    const description = screen.getByText(
      "Enter email and password for your account"
    );
    const emailInput = screen.getByPlaceholderText("Email Address");
    const passwordInput = screen.getByPlaceholderText("Password");
    const confirmPasswordInput = screen.getByPlaceholderText("Confirm Password");
    const createButton = screen.getByText("Create");

    expect(title).toBeInTheDocument();
    expect(description).toBeInTheDocument();
    expect(emailInput).toBeInTheDocument();
    expect(passwordInput).toBeInTheDocument();
    expect(confirmPasswordInput).toBeInTheDocument();
    expect(createButton).toBeInTheDocument();
  });

  test("handles email and password input", async () => {
    const emailInput = screen.getByPlaceholderText("Email Address");
    const passwordInput = screen.getByPlaceholderText("Password");
    const confirmPasswordInput = screen.getByPlaceholderText("Confirm Password");

    await act(async () => {
      fireEvent.change(emailInput, { target: { value: "test@example.com" } });
      fireEvent.change(passwordInput, { target: { value: "Password123!" } });
      fireEvent.change(confirmPasswordInput, { target: { value: "Password123!" } });
    });

    expect(emailInput).toHaveValue("test@example.com");
    expect(passwordInput).toHaveValue("Password123!");
    expect(confirmPasswordInput).toHaveValue("Password123!");
  });

  test("calls signUpNewUser on form submission", async () => {
    const emailInput = screen.getByPlaceholderText("Email Address");
    const passwordInput = screen.getByPlaceholderText("Password");
    const confirmPasswordInput = screen.getByPlaceholderText("Confirm Password");
    const createButton = screen.getByText("Create");

    await act(async () => {
      fireEvent.change(emailInput, { target: { value: "test@example.com" } });
      fireEvent.change(passwordInput, { target: { value: "Password123!" } });
      fireEvent.change(confirmPasswordInput, { target: { value: "Password123!" } });
      fireEvent.click(createButton);
    });

    expect(mockSignUpNewUser).toHaveBeenCalledTimes(1);
    expect(mockSignUpNewUser).toHaveBeenCalledWith(
      "test@example.com",
      "Password123!"
    );
  });

  test("validates password requirements", async () => {
    const passwordInput = screen.getByPlaceholderText("Password");

    await act(async () => {
      fireEvent.change(passwordInput, { target: { value: "short" } });
    });

    expect(screen.getByText("❌ At least 8 characters")).toBeInTheDocument();
    expect(screen.getByText("❌ At least 1 uppercase letter")).toBeInTheDocument();
    expect(screen.getByText("❌ At least 1 number")).toBeInTheDocument();
    expect(screen.getByText("❌ At least 1 special character")).toBeInTheDocument();

    await act(async () => {
      fireEvent.change(passwordInput, { target: { value: "Password123!" } });
    });

    expect(screen.getByText("✅ At least 8 characters")).toBeInTheDocument();
    expect(screen.getByText("✅ At least 1 uppercase letter")).toBeInTheDocument();
    expect(screen.getByText("✅ At least 1 number")).toBeInTheDocument();
    expect(screen.getByText("✅ At least 1 special character")).toBeInTheDocument();
  });


  test("toggles password visibility", async () => {
    const passwordInput = screen.getByPlaceholderText("Password");
    const toggleVisibilityButtons = screen.getAllByText("Eye Icon");
    const toggleVisibilityButton = toggleVisibilityButtons[0];

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


});