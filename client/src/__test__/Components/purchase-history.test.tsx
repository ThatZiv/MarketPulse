import { render, screen, waitFor } from "@testing-library/react";
import PurchaseHistory from "@/components/purchase-history";
import { useSupabase } from "@/database/SupabaseProvider";
import { useGlobal } from "@/lib/GlobalProvider";
import "@testing-library/jest-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import moment from "moment";
import { actions } from "@/lib/constants";

// Mocking dependencies
jest.mock("@/database/SupabaseProvider", () => ({
  useSupabase: jest.fn(),
}));

jest.mock("@/lib/GlobalProvider", () => ({
  useGlobal: jest.fn(),
}));

jest.mock("lucide-react", () => ({
  Check: () => "CheckIcon",
  ChevronDown: () => "ChevronDownIcon",
  ChevronUp: () => "ChevronUpIcon",
}));

window.ResizeObserver =
  window.ResizeObserver ||
  jest.fn().mockImplementation(() => ({
    disconnect: jest.fn(),
    observe: jest.fn(),
    unobserve: jest.fn(),
  }));

const queryClient = new QueryClient();

describe("PurchaseHistory Component", () => {
  const mockDispatch = jest.fn();
  const mockSupabase = {
    from: jest.fn().mockReturnThis(),
    select: jest.fn().mockReturnThis(),
    eq: jest.fn().mockReturnThis(),
    order: jest.fn().mockReturnThis(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    (useSupabase as jest.Mock).mockReturnValue({
      supabase: mockSupabase,
      user: { id: 123 },
    });

    (useGlobal as jest.Mock).mockReturnValue({
      dispatch: mockDispatch,
    });

    queryClient.clear();
  });

  it("should render purchase data correctly", async () => {
    const mockData = [
      {
        date: new Date(),
        amount_purchased: 10,
        price_purchased: 150,
      },
    ];

    mockSupabase.order.mockResolvedValue({ data: mockData, error: null });

    render(
      <QueryClientProvider client={queryClient}>
        <PurchaseHistory ticker="AAPL" stock_id={1} />
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(mockSupabase.from).toHaveBeenCalledWith("User_Stock_Purchases");
      expect(mockSupabase.select).toHaveBeenCalledWith("*");
      expect(mockSupabase.eq).toHaveBeenCalledWith("stock_id", 1);
      expect(mockSupabase.eq).toHaveBeenCalledWith("user_id", 123);
      expect(mockDispatch).toHaveBeenCalledWith(
        expect.objectContaining({
          type: actions.SET_USER_STOCK_TRANSACTIONS,
          payload: {
            stock_ticker: "AAPL",
            data: mockData,
          },
        })
      );
      const prices = screen.getAllByText(/\$150/);

      for (const price of prices) {
        expect(price).toBeInTheDocument();
      }

      expect(screen.getByText("$1,500.00")).toBeInTheDocument();
      expect(
        screen.getByText(moment(mockData[0].date).format("MMMM DD, yyyy"))
      ).toBeInTheDocument();
    });
  });

  it("should handle error state", async () => {
    mockSupabase.order.mockResolvedValue({ data: null, error: "Test Error" });

    render(
      <QueryClientProvider client={queryClient}>
        <PurchaseHistory ticker="AAPL" stock_id={1} />
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(
        screen.getByText(/no purchase history available/i)
      ).toBeInTheDocument();
    });
  });
});
