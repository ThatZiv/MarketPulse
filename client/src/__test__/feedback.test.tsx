import Feedback from "@/pages/Feedback";
import { describe, test, afterEach, beforeEach } from "@jest/globals";
import {
  render,
  screen,
  fireEvent,
  act,
  cleanup,
} from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import "@testing-library/jest-dom";

jest.mock("sonner", () => ({
    toast: {
      success: jest.fn(),
      error: jest.fn(),
    },
  }));

const mockInsertFeedback = jest.fn();
const mockNavigate = jest.fn();

jest.mock("@/database/SupabaseProvider", () => ({
  useSupabase: () => ({
    supabase: {
      from: () => ({
        insert: mockInsertFeedback,
      }),
    },
  }),
}));

jest.mock("react-router-dom", () => ({
  ...jest.requireActual("react-router-dom"),
  useNavigate: () => mockNavigate,
}));

afterEach(() => {
  cleanup();
  jest.clearAllMocks();
});

describe("Feedback Page", () => {
  beforeEach(() => {
    render(
      <MemoryRouter>
        <Feedback />
      </MemoryRouter>
    );
  });

  test("renders Feedback page correctly", () => {
    expect(screen.getByText("Feedback")).toBeInTheDocument();
    expect(
      screen.getByText(/We value your feedback! Please share your thoughts/i)
    ).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText("Enter your feedback here...")
    ).toBeInTheDocument();
    expect(screen.getByText("Submit Feedback")).toBeInTheDocument();
  });

  test("handles feedback input", async () => {
    const feedbackInput = screen.getByPlaceholderText(
      "Enter your feedback here..."
    );

    await act(async () => {
      fireEvent.change(feedbackInput, {
        target: { value: "This is a test feedback." },
      });
    });

    expect(feedbackInput).toHaveValue("This is a test feedback.");
  });


  test("validates empty feedback submission", async () => {
    const submitButton = screen.getByText("Submit Feedback");

    await act(async () => {
      fireEvent.click(submitButton);
    });

    expect(mockInsertFeedback).not.toHaveBeenCalled();
    expect(require("sonner").toast.error).toHaveBeenCalledWith(
      "Feedback cannot be empty!"
    );
  });



});