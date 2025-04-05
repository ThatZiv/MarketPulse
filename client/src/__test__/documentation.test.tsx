import Documentation from "@/pages/Documentation";
import { describe, test, afterEach, beforeEach } from "@jest/globals";
import {
  render,
  screen,
  fireEvent,
  cleanup,
} from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import "@testing-library/jest-dom";


jest.mock("react-router-dom", () => ({
    ...jest.requireActual("react-router-dom"),
    Link: ({ to, children }: { to: string; children: React.ReactNode }) => (
      <a href={to}>{children}</a>
    ),
  }));


  afterEach(() => {
    cleanup();
  });
  
  describe("Documentation Page", () => {
    beforeEach(() => {
      render(
        <MemoryRouter>
          <Documentation />
        </MemoryRouter>
      );
    });

    test("renders Documentation page correctly", () => {
        expect(screen.getByText("Documentation")).toBeInTheDocument();
        expect(screen.getByText("Introduction")).toBeInTheDocument();
        expect(screen.getByText("Tutorials")).toBeInTheDocument();
        expect(screen.getByText("FAQ")).toBeInTheDocument();
        expect(screen.getByText("Disclaimer")).toBeInTheDocument();
      });


      test("navigates to Introduction page when Introduction button is clicked", () => {
        const introductionLink = screen.getByText("Introduction");
        expect(introductionLink).toHaveAttribute("href", "/documentation/introduction");
    
        fireEvent.click(introductionLink);
      });


      test("navigates to Tutorials page when Tutorials button is clicked", () => {
        const tutorialsLink = screen.getByText("Tutorials");
        expect(tutorialsLink).toHaveAttribute("href", "/documentation/tutorials");
    
        fireEvent.click(tutorialsLink);
      });


      test("navigates to FAQ page when FAQ button is clicked", () => {
        const faqLink = screen.getByText("FAQ");
        expect(faqLink).toHaveAttribute("href", "/documentation/faq");
    
        fireEvent.click(faqLink);
      });


      test("navigates to Disclaimer page when Disclaimer button is clicked", () => {
        const disclaimerLink = screen.getByText("Disclaimer");
        expect(disclaimerLink).toHaveAttribute("href", "/documentation/disclaimer");
    
        fireEvent.click(disclaimerLink);
      });






});