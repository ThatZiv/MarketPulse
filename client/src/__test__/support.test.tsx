import Support from "@/pages/Support";
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


  jest.mock("lucide-react", () => ({
    LockIcon: () => <span>Lock Icon</span>,
    SaveIcon: () => <span>Save Icon</span>,
    Eye: () => <span>Eye Icon</span>,
    EyeOff: () => <span>EyeOff Icon</span>,
  }));


const mockInsertSupportRequest = jest.fn();
const mockNavigate = jest.fn();

jest.mock("@/database/SupabaseProvider", () => ({
  useSupabase: () => ({
    supabase: {
      from: () => ({
        insert: mockInsertSupportRequest,
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
  
  describe("Support Page", () => {
    beforeEach(() => {
      render(
        <MemoryRouter>
          <Support />
        </MemoryRouter>
      );
    });
  
    test("renders Support page correctly", () => {
      expect(screen.getByText("Support")).toBeInTheDocument();
      expect(
        screen.getByText(
          /Need help\? Please select the type of issue you're facing and provide a brief summary/i
        )
      ).toBeInTheDocument();
      expect(screen.getByLabelText("Issue Type")).toBeInTheDocument();
      expect(screen.getByLabelText("Summary")).toBeInTheDocument();
      expect(screen.getByText("Submit Request")).toBeInTheDocument();
    });

    test("handles issue type and summary input", async () => {
        const issueTypeSelect = screen.getByLabelText("Issue Type");
        const summaryTextarea = screen.getByLabelText("Summary");
    
        await act(async () => {
          fireEvent.change(issueTypeSelect, { target: { value: "Bug" } });
          fireEvent.change(summaryTextarea, {
            target: { value: "This is a test summary." },
          });
        });
    
        expect(issueTypeSelect).toHaveValue("Bug");
        expect(summaryTextarea).toHaveValue("This is a test summary.");
      });


      test("validates empty form submission", async () => {
        const submitButton = screen.getByText("Submit Request");
    
        await act(async () => {
          fireEvent.click(submitButton);
        });
    
        expect(mockInsertSupportRequest).not.toHaveBeenCalled();
        expect(toast.error).toHaveBeenCalledWith(
          "Both issue type and summary are required!"
        );
      });



      test("submits support request successfully", async () => {
        const issueTypeSelect = screen.getByLabelText("Issue Type");
        const summaryTextarea = screen.getByLabelText("Summary");
        const submitButton = screen.getByText("Submit Request");
    
        mockInsertSupportRequest.mockResolvedValueOnce({ error: null });
    
        await act(async () => {
          fireEvent.change(issueTypeSelect, { target: { value: "Bug" } });
          fireEvent.change(summaryTextarea, {
            target: { value: "This is a test summary." },
          });
          fireEvent.click(submitButton);
        });
    
        expect(mockInsertSupportRequest).toHaveBeenCalledTimes(1);
        expect(mockInsertSupportRequest).toHaveBeenCalledWith([
          { issue_type: "Bug", summary: "This is a test summary." },
        ]);
        expect(toast.success).toHaveBeenCalledWith(
          "Your support request has been submitted!"
        );
        expect(mockNavigate).toHaveBeenCalledWith("/");
      });
    

      test("handles support request submission error", async () => {
        const issueTypeSelect = screen.getByLabelText("Issue Type");
        const summaryTextarea = screen.getByLabelText("Summary");
        const submitButton = screen.getByText("Submit Request");
    
        const errorMessage = "An unexpected error occurred";
    
        mockInsertSupportRequest.mockResolvedValueOnce({
          error: { message: errorMessage },
        });
    
        await act(async () => {
          fireEvent.change(issueTypeSelect, { target: { value: "Bug" } });
          fireEvent.change(summaryTextarea, {
            target: { value: "This is a test summary." },
          });
          fireEvent.click(submitButton);
        });
    
        expect(mockInsertSupportRequest).toHaveBeenCalledTimes(1);
        expect(mockInsertSupportRequest).toHaveBeenCalledWith([
          { issue_type: "Bug", summary: "This is a test summary." },
        ]);
        expect(toast.error).toHaveBeenCalledWith(errorMessage);
      });


});