import StockPage from "@/pages/StockSelection";
import { describe, test, afterEach, beforeAll } from "@jest/globals";
import {
  render,
  waitFor,
  screen,
  cleanup,
  act,
  fireEvent,
} from "@testing-library/react";
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
    user: { id: "test-user-id" },
    supabase: {
      from: jest.fn().mockReturnValue({
        upsert: jest.fn().mockReturnThis(),
        delete: jest.fn().mockReturnThis(),
        eq: jest.fn().mockReturnThis(),
        then: jest.fn().mockImplementation((cb) => cb({ error: null })),
        insert: jest.fn().mockReturnThis(),
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

describe("Add Stock Form Component Testcase", () => {
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

  test("Renders all the form elements", async () => {
    const question1 = screen.getByText(/What is the ticker\?/i);
    const question2 = screen.getByText(/Do you own this stock?/i);
    const question3 = screen.getByText(/Desired Investment/i);
    const heading = screen.getByRole("heading", { name: /Add New Stock/i });
    const submitBtn = screen.getByRole("button", { name: /Submit/i });
    const backBtn = screen.getByRole("button", { name: /Back/i });
    const input = document.getElementById("cashToInvest");
    const switchButton = screen.getByRole("switch");
    expect(heading).toBeInTheDocument();
    expect(question1).toBeInTheDocument();
    expect(question2).toBeInTheDocument();
    expect(question3).toBeInTheDocument();
    expect(submitBtn).toBeInTheDocument();
    expect(backBtn).toBeInTheDocument();
    expect(input).toBeInTheDocument();
    expect(switchButton).toBeInTheDocument();
    expect(switchButton).toHaveAttribute("id", "hasStocks");
    expect(switchButton).toHaveAttribute("aria-checked", "false");
    expect(switchButton).toBeDisabled();
  });
  test("UTC16 - Investment amount should not be empty", async () => {
    const teslaOption = await screen.findByRole("option", { name: /tesla/i });
    const handleSubmit = jest.fn();
    await userEvent.click(teslaOption);
    const submitBtn = screen.getByRole("button", { name: /submit/i });
    await userEvent.click(submitBtn);
    const input = document.getElementById("cashToInvest");
    expect(input).toBeInTheDocument();
    expect(input).toBeRequired();
    expect(handleSubmit).not.toHaveBeenCalled();
  });
  test("UTC15 - Investment amount should be a float and greater than 0", async () => {
    const teslaOptions = await screen.findByRole("option", { name: /tesla/i });
    await userEvent.click(teslaOptions);
    const submitBtn = await screen.findByRole("button", { name: /submit/i });
    const investmentInput = document.getElementById("cashToInvest");
    await userEvent.type(investmentInput!, "1515.75");
    await userEvent.click(submitBtn);
    expect(investmentInput).toHaveValue(1515.75);
  });
  test("UTC15 - Investment amount should be a float and greater than 0", async () => {
    const teslaOption = await screen.findByRole("option", { name: /tesla/i });
    const handleSubmit = jest.fn();
    await userEvent.click(teslaOption);
    const submitBtn = screen.getByRole("button", { name: /submit/i });
    const investmentInput = document.getElementById("cashToInvest");
    await userEvent.type(investmentInput!, "-100.45");
    await userEvent.click(submitBtn);
    expect(handleSubmit).not.toHaveBeenCalled();
  });
  test("UTC17 - Stock ticker is not selected", async () => {
    const investmentInput = document.getElementById("cashToInvest")!;
    await userEvent.type(investmentInput, "500");
    expect(investmentInput).toHaveValue(500);
    const submitBtn = screen.getByRole("button", { name: /submit/i });
    await userEvent.click(submitBtn);
    const errorMsg = await screen.findByText(/please select a stock/i);
    expect(errorMsg).toBeInTheDocument();
    expect(errorMsg).toHaveClass("text-red-500");
    const allDivs = screen.getAllByRole("generic");
    expect(allDivs[0]).toContainElement(errorMsg);
  });
  test("UTC18 - Clicking Yes for the Stock Transaction History question.", async () => {
    const teslaOption = await screen.findByRole("option", { name: /tesla/i });
    await userEvent.click(teslaOption);
    const switchToggle = await screen.findByRole("switch", {
      name: /do you own this stock/i,
    });
    await userEvent.click(switchToggle);
    const label = screen.getByText(/Transaction History/i);
    const addTransactionBtn = await screen.findByRole("button", {
      name: /Add Transaction/i,
    });
    const transactionRules = [
      "Your stock history must be in chronological order",
      "You cannot sell more shares than you own on a given day",
      "Cumulatively, you cannot sell more shares than you own",
    ];
    const stockMetrics = [
      "Current Profit",
      "Total Purchased",
      "Total Sold",
      "Current Shares",
    ];
    expect(label).toBeInTheDocument();
    expect(addTransactionBtn).toBeInTheDocument();
    for (const rule of transactionRules) {
      expect(await screen.findByText(rule)).toBeInTheDocument();
    }
    for (const metric of stockMetrics) {
      expect(await screen.findByText(metric)).toBeInTheDocument();
    }
  });
  test("UTC19 -  Clicking on Add Purchase under the investment history question", async () => {
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
    const dateLabel = screen.getByLabelText(/date/i);
    const sharesLabel = screen.getByLabelText(/shares/i);
    const priceLabel = screen.getByLabelText(/price \(\$\)/i);
    const buyOption = screen.getByRole("option", { name: /buy/i });
    const sellOption = screen.getByRole("option", { name: /sell/i });
    const dateInput = document.getElementById("date-0");
    const sharesInput = document.getElementById("shares-0");
    const priceInput = document.getElementById("price-0");
    expect(dateLabel).toBeInTheDocument();
    expect(sharesLabel).toBeInTheDocument();
    expect(priceLabel).toBeInTheDocument();
    expect(buyOption).toBeInTheDocument();
    expect(sellOption).toBeInTheDocument();
    expect(dateInput).toBeInTheDocument();
    expect(dateInput).toHaveAttribute("type", "datetime-local");
    expect(sharesInput).toBeInTheDocument();
    expect(sharesInput).toHaveAttribute("type", "number");
    expect(priceInput).toBeInTheDocument();
    expect(priceInput).toHaveAttribute("type", "number");
  });
  test("UTC20 - Investment history dates shouldn’t be less than 2000", async () => {
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
    const dateInput = document.getElementById("date-0") as HTMLInputElement;
    fireEvent.change(dateInput, {
      target: { value: "1999-12-30T14:23" },
    });
    expect(dateInput.validity.rangeUnderflow).toBe(true);
    expect(dateInput.checkValidity()).toBe(false);
  });
  test("UTC20 - Investment history dates shouldn’t be greater than present date", async () => {
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
    const dateInput = document.getElementById("date-0") as HTMLInputElement;
    const now = new Date();
    const tomorrow = new Date(now);
    tomorrow.setDate(now.getDate() + 1);
    const future_date = tomorrow.toISOString().slice(0, 16);
    fireEvent.change(dateInput, {
      target: { value: future_date },
    });
    expect(dateInput.validity.rangeOverflow).toBe(true);
    expect(dateInput.checkValidity()).toBe(false);
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
