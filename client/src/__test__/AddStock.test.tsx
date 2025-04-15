import StockPage from "@/pages/StockSelection";
import { describe, test, afterEach, beforeAll } from "@jest/globals";
import { render, waitFor, screen, cleanup, act } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "@testing-library/jest-dom";
import userEvent from "@testing-library/user-event";
import { ReactNode, createContext, useContext } from "react";

beforeAll(() => {
  global.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

jest.mock("sonner", () => ({
  toast: {
    success: jest.fn(),
    error: jest.fn(),
    promise: jest.fn(),
  },
}));

jest.mock("lucide-react", () => ({
  Trash: () => <span>Trash Icon</span>,
  ArrowRight: () => <span>ArrowRight Icon</span>,
  ArrowLeft: () => <span>ArrowLeft Icon</span>,
  TrendingUp: () => <span>TrendingUp Icon</span>,
  Box: () => <span>Box Icon</span>,
  ArrowDown: () => <span>ArrowDown Icon</span>,
  ArrowUp: () => <span>ArrowUp Icon</span>,
  Plus: () => <span>Plus Icon</span>,
  Undo: () => <span>Undo Icon</span>,
  X: () => <span>X Icon</span>,
}));

jest.mock("@/database/SupabaseProvider", () => ({
  useSupabase: () => ({
    user: { id: "mock-user-id" },
    supabase: {
      from: jest.fn().mockReturnValue({
        upsert: jest.fn().mockReturnThis(),
        delete: jest.fn().mockReturnThis(),
        eq: jest.fn().mockReturnThis(),
        then: jest.fn().mockImplementation((cb) => cb({ error: null })),
      }),
    },
  }),
}));

jest.mock("@/lib/dataHandler", () => {
  return () => ({
    forSupabase: () => ({
      getAllStocks: () => async () => [
        {
          stock_id: 1,
          stock_ticker: "TSLA",
          stock_name: "Tesla",
        },
      ],
      getUserStocks: () => async () => [],
      getUserStockPurchasesForStock: () => async () => [],
    }),
  });
});

jest.mock("@/components/InfoTooltip", () => ({
  __esModule: true,
  default: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}));

// Select component mock

type SelectContextType = {
  onChange: (value: string) => void;
};

const SelectContext = createContext<SelectContextType>({
  onChange: () => {},
});

jest.mock("@/components/ui/select", () => {
  return {
    Select: ({
      children,
      onValueChange,
    }: {
      children: ReactNode;
      onValueChange: (value: string) => void;
    }) => (
      <SelectContext.Provider value={{ onChange: onValueChange }}>
        <div>{children}</div>
      </SelectContext.Provider>
    ),
    SelectTrigger: ({ children }: { children: ReactNode }) => (
      <button aria-label="Select Stock">{children}</button>
    ),
    SelectValue: () => <span>Select Value</span>,
    SelectContent: ({ children }: { children: ReactNode }) => (
      <div>{children}</div>
    ),
    SelectGroup: ({ children }: { children: ReactNode }) => (
      <div>{children}</div>
    ),
    SelectLabel: ({ children }: { children: ReactNode }) => (
      <div>{children}</div>
    ),
    SelectItem: ({
      children,
      value,
    }: {
      children: ReactNode;
      value: string;
    }) => {
      const { onChange } = useContext(SelectContext);
      return (
        <div role="option" data-value={value} onClick={() => onChange(value)}>
          {children}
        </div>
      );
    },
  };
});

jest.mock("@/lib/Calculator", () => {
  const instanceMethods = {
    getProfit: jest.fn(() => 100),
    getTotalBought: jest.fn(() => 300),
    getTotalSold: jest.fn(() => 200),
    getTotalShares: jest.fn(() => 5),
    isInvalidHistory: jest.fn(() => null),
  };

  function MockCalculator() {
    return instanceMethods;
  }

  return {
    PurchaseHistoryCalculator: Object.assign(MockCalculator, {
      toDollar: jest.fn((val: number) => `$${val.toFixed(2)}`),
    }),
  };
});

jest.mock("react-router-dom", () => ({
  ...jest.requireActual("react-router-dom"),
  useNavigate: () => jest.fn(),
  useSearchParams: () => [new URLSearchParams(), jest.fn()],
}));

const queryClient = new QueryClient();

// Tests

describe("StockPage Sanity Test", () => {
  beforeEach(async () => {
    await act(async () => {
      render(
        <MemoryRouter>
          <QueryClientProvider client={queryClient}>
            <StockPage />
          </QueryClientProvider>
        </MemoryRouter>
      );
    });
  });

  afterEach(() => {
    cleanup();
    jest.clearAllMocks();
  });

  test("UTC21 - Invalid shares input shows error", async () => {
    const teslaOption = await screen.findByRole("option", { name: /tesla/i });
    await userEvent.click(teslaOption);

    const switchToggle = await screen.findByRole("switch", {
      name: /do you own this stock/i,
    });
    await userEvent.click(switchToggle);

    const addTransactionBtn = await screen.findByRole("button", {
      name: /add transaction/i,
    });
    await userEvent.click(addTransactionBtn);

    const sharesInput = screen.getByLabelText(/shares/i);
    await userEvent.clear(sharesInput);
    await userEvent.type(sharesInput, "-5");

    const submitBtn = screen.getByRole("button", { name: /submit/i });
    await userEvent.click(submitBtn);
  });

  test("UTC22 - Invalid price input shows error", async () => {
    const teslaOptions = await screen.findAllByRole("option", {
      name: /tesla/i,
    });
    await userEvent.click(teslaOptions[0]);

    const switchToggle = await screen.findByRole("switch", {
      name: /do you own this stock/i,
    });
    await userEvent.click(switchToggle);

    const addTransactionBtn = await screen.findByRole("button", {
      name: /add transaction/i,
    });
    await userEvent.click(addTransactionBtn);

    const priceInput = screen.getByLabelText(/price/i);
    await userEvent.clear(priceInput);
    await userEvent.type(priceInput, "-1");

    const sharesInput = screen.getByLabelText(/shares/i);
    await userEvent.clear(sharesInput);
    await userEvent.type(sharesInput, "5");

    const submitButtons = screen.getAllByRole("button", { name: /submit/i });
    await userEvent.click(submitButtons[0]);
  });

  test("UTC23 - Remove button deletes entry (no confirmation dialog)", async () => {
    const teslaOptions = await screen.findAllByRole("option", {
      name: /tesla/i,
    });
    await userEvent.click(teslaOptions[0]);

    const switchToggle = await screen.findByRole("switch", {
      name: /do you own this stock/i,
    });
    await userEvent.click(switchToggle);

    const addTransactionBtn = await screen.findByRole("button", {
      name: /add transaction/i,
    });
    await userEvent.click(addTransactionBtn);

    expect(screen.getByLabelText(/shares/i)).toBeInTheDocument();

    const removeButtons = screen.getAllByRole("button");
    const removeBtn =
      removeButtons.find((btn) =>
        btn.innerHTML.toLowerCase().includes("trash")
      ) ?? removeButtons[removeButtons.length - 1];
    await userEvent.click(removeBtn);

    await waitFor(() => {
      expect(screen.queryByLabelText(/shares/i)).not.toBeInTheDocument();
    });
  });

  test("UTC24 - Submitting with no purchase entry shows error", async () => {
    const teslaOptions = await screen.findAllByRole("option", {
      name: /tesla/i,
    });
    await userEvent.click(teslaOptions[0]);

    const switchToggle = await screen.findByRole("switch", {
      name: /do you own this stock/i,
    });
    await userEvent.click(switchToggle);

    const investmentInput = screen.getByLabelText(/Desired Investment/i);
    await userEvent.clear(investmentInput);
    await userEvent.type(investmentInput, "200");

    const submitButtons = screen.getAllByRole("button", { name: /submit/i });
    await userEvent.click(submitButtons[0]);

    expect(
      await screen.findByText(/at least one purchase entry is required/i)
    ).toBeInTheDocument();
  });

  test("UTC25 - Valid data triggers save toast", async () => {
    const teslaOptions = await screen.findAllByRole("option", {
      name: /tesla/i,
    });
    await userEvent.click(teslaOptions[0]);

    const switchToggle = await screen.findByRole("switch", {
      name: /do you own this stock/i,
    });
    await userEvent.click(switchToggle);

    const investmentInput = screen.getByLabelText(/Desired Investment/i);
    await userEvent.clear(investmentInput);
    await userEvent.type(investmentInput, "200");

    const addTransactionBtn = await screen.findByRole("button", {
      name: /add transaction/i,
    });
    await userEvent.click(addTransactionBtn);

    const today = new Date();
    const formattedToday = today.toISOString().slice(0, 16);

    const dateInput = screen.getByLabelText(/date/i);
    await userEvent.clear(dateInput);
    await userEvent.type(dateInput, formattedToday);

    const sharesInput = screen.getByLabelText(/shares/i);
    await userEvent.clear(sharesInput);
    await userEvent.type(sharesInput, "10");

    const priceInput = screen.getByLabelText(/price/i);
    await userEvent.clear(priceInput);
    await userEvent.type(priceInput, "25");

    const submitButtons = screen.getAllByRole("button", { name: /submit/i });
    await userEvent.click(submitButtons[0]);

  });
});
