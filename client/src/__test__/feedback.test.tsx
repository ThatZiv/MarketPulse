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
import { toast } from "sonner";

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
    expect(toast.error).toHaveBeenCalledWith("Feedback cannot be empty!");
  });


  test("submits feedback successfully", async () => {
    const feedbackInput = screen.getByPlaceholderText(
      "Enter your feedback here..."
    );
    const submitButton = screen.getByText("Submit Feedback");

    mockInsertFeedback.mockResolvedValueOnce({ error: null });

    await act(async () => {
      fireEvent.change(feedbackInput, {
        target: { value: "This is a test feedback." },
      });
      fireEvent.click(submitButton);
    });

    expect(mockInsertFeedback).toHaveBeenCalledTimes(1);
    expect(mockInsertFeedback).toHaveBeenCalledWith([
      { content: "This is a test feedback." },
    ]);
    expect(toast.success).toHaveBeenCalledWith("Thank you for your feedback!");
    expect(mockNavigate).toHaveBeenCalledWith("/");
  });



  test("handles feedback submission error", async () => {
    const feedbackInput = screen.getByPlaceholderText(
      "Enter your feedback here..."
    );
    const submitButton = screen.getByText("Submit Feedback");

    const errorMessage = "An unexpected error occurred";

    mockInsertFeedback.mockResolvedValueOnce({
      error: { message: errorMessage },
    });

    await act(async () => {
      fireEvent.change(feedbackInput, {
        target: { value: "This is a test feedback." },
      });
      fireEvent.click(submitButton);
    });

    expect(mockInsertFeedback).toHaveBeenCalledTimes(1);
    expect(mockInsertFeedback).toHaveBeenCalledWith([
      { content: "This is a test feedback." },
    ]);
    expect(toast.error).toHaveBeenCalledWith(errorMessage); 
   });



  test("disables submit button while loading", async () => {
    const feedbackInput = screen.getByPlaceholderText(
      "Enter your feedback here..."
    );
    const submitButton = screen.getByText("Submit Feedback");

    mockInsertFeedback.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          setTimeout(() => resolve({ error: null }), 1000);
        })
    );

    await act(async () => {
      fireEvent.change(feedbackInput, {
        target: { value: "This is a test feedback." },
      });
      fireEvent.click(submitButton);
    });

    expect(submitButton).toBeDisabled();
  });



});