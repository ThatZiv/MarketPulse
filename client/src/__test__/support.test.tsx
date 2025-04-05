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