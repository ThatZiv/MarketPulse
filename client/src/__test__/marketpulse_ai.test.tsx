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
  Loader2: () => "Loader2",
}));

jest.mock("@/lib/ApiProvider", () => ({
  useApi: () => {
    return {
      getStockLlmOutput: jest.fn((ticker, callBack) => {
        console.log(ticker);
        callBack("</think>Hello this is the llm stream");
      }),
    };
  },
}));

jest.mock("react-router", () => ({
  Link: jest.fn(),
}));

describe("MarketPulse AI", () => {
  beforeAll(() => {
    global.matchMedia = jest.fn().mockImplementation((query) => ({
      matches: false,
      media: query,
      addListener: jest.fn(),
      removeListener: jest.fn(),
    }));
  });

  test("Render", async () => {
    const ticker = "test";
    await render(<GenerateStockLLM ticker={ticker} />);
    const open = await screen.findByText("MarketPulse AI");
    expect(open).toBeInTheDocument();
    expect(open).not.toBeDisabled();

    expect(open).toBeVisible();

    await act(async () => {
      await fireEvent.click(open);
    });

    const close = await screen.findByText("Close");
    expect(close).toBeInTheDocument();

    const stream = await screen.findByText("Hello this is the llm stream");
    expect(stream).toBeInTheDocument();

    //mockUseMemo.mockRestore();
  });
});
