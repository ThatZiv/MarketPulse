import { CreateForm } from "@/components/create_user_form";
import { describe, test, afterEach, beforeEach } from "@jest/globals";
import {
  render,
  screen,

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

});