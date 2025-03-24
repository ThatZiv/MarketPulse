
import { describe } from "node:test";
import {
  act,
  render,
  screen,
  cleanup,
  fireEvent,
} from "@testing-library/react";

import "@testing-library/jest-dom";

import { GenerateStockLLM } from "@/components/llm/stock-llm";



afterEach(() => {
  jest.restoreAllMocks();
  cleanup();
});


jest.mock("lucide-react", () => ({
    Moon: () => "MoonIcon",
    Sun: () => "SunIcon",
    Eye: () => "EyeIcon",
    EyeOff: () => "EyeOffIcon",
    ChevronDown: () => "ChevronDown",
    Loader2: () => "Loader2"
  }));

jest.mock("@/lib/ApiProvider", () => ({
    useApi: () => {
            return {getStockLlmOutput: jest.fn(() => {return "</think>Hello"})}}
    } 
))

jest.mock("react-router", () => ({
  Link: jest.fn()
}))

//jest.mock("@/components/ui/accordion", () => ({
//  Accordion: jest.fn(),
//  AccordionContent: jest.fn(),
//  AccordionItem: jest.fn(),
//  AccordionTrigger: jest.fn(),
//}))


//import axios, { type AxiosInstance, type AxiosError } from "axios";



describe("MarketPulse AI", () => {
  beforeAll(() => {
    global.matchMedia = jest.fn().mockImplementation(query => ({
      matches: false, 
      media: query,
      addListener: jest.fn(),
      removeListener: jest.fn(),}));

  });
  
    test("Render", async () => {
        const ticker = "test"
        render(<GenerateStockLLM ticker={ticker} />);
        const open = await screen.findByText("MarketPulse AI");
        expect(open).toBeInTheDocument();
        expect(open).not.toBeDisabled();

        expect(open).toBeVisible();

        await act(async () => {
          await fireEvent.click(open);
        });

        const close = await screen.findByText("Close");
        expect(close).toBeInTheDocument();

        const think = await screen.findByText("Thinking...ChevronDown");
        expect(think).toBeInTheDocument();

        await act(async () => {
          await fireEvent.click(think);
        });
        
        const text_test = await screen.findByText("Hello");
      });
});