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





});