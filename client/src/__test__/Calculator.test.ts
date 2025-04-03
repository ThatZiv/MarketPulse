import { PurchaseHistoryCalculator } from "@/lib/Calculator";
import { PurchaseHistoryDatapoint } from "@/types/global_state";
import "@testing-library/jest-dom";

describe("Purchase History Calculator", () => {
  const calc = new PurchaseHistoryCalculator([]);
  const purchases = [
    { date: "2025-03-01", amount_purchased: 10, price_purchased: 10 },
    { date: "2025-03-02", amount_purchased: 20, price_purchased: 20 },
    { date: "2025-03-03", amount_purchased: 30, price_purchased: 30 },
  ] as unknown as PurchaseHistoryDatapoint[];
  const purchases2 = [
    { date: "2025-03-01", amount_purchased: 10, price_purchased: 10 },
    { date: "2025-03-02", amount_purchased: -20, price_purchased: 20 },
    { date: "2025-03-03", amount_purchased: 30, price_purchased: 30 },
  ] as unknown as PurchaseHistoryDatapoint[];

  const purchases3 = [
    { date: "2025-03-01", amount_purchased: 10, price_purchased: 10 },
    { date: "2025-03-02", amount_purchased: -20, price_purchased: 20 },
    { date: "2025-03-03", amount_purchased: -30, price_purchased: 30 },
  ] as unknown as PurchaseHistoryDatapoint[];

  test("should calculate total(s)", () => {
    calc.setPurchases(purchases);
    expect(calc.getTotalShares()).toBe(60);

    expect(calc.getTotalBought()).toBe(1400);

    expect(calc.getTotalSold()).toBe(0);

    calc.setPurchases(purchases2);
    expect(calc.getTotalShares()).toBe(20);
    expect(calc.getTotalBought()).toBe(1000);
    expect(calc.getTotalSold()).toBe(400);

    calc.setPurchases(purchases3);

    expect(calc.getTotalShares()).toBe(-40);
    expect(calc.getTotalBought()).toBe(100);

    expect(calc.getTotalSold()).toBe(1300);
  });

  test("should calculate profit", () => {
    calc.setPurchases(purchases);

    expect(calc.getProfit()).toBe(0);

    calc.setPurchases(purchases2);
    // cant sell more than you have
    expect(calc.isInvalidHistory()).toBe("2025-03-02");
    expect(calc.getProfit()).toBe(200);

    calc.setPurchases(purchases3);
    // cant sell more than you have
    expect(calc.isInvalidHistory()).toBe("2025-03-02");
    expect(calc.getProfit()).toBe(600);
  });

  test("should calculate average price", () => {
    calc.setPurchases(purchases);
    expect(calc.getAveragePrice()).toBe(23.333333333333332);

    calc.setPurchases(purchases2);
    expect(calc.getAveragePrice()).toBe(40);
  });

  test("should calculate total value", () => {
    const currentPrice = 10;
    calc.setPurchases(purchases);
    expect(calc.getTotalValue(currentPrice)).toBe(600);

    calc.setPurchases(purchases2);
    expect(calc.getTotalValue(currentPrice)).toBe(200);
  });

  test("should convert to dollar", () => {
    expect(PurchaseHistoryCalculator.toDollar(10)).toBe("$10.00");
    expect(PurchaseHistoryCalculator.toDollar(-10)).toBe("-$10.00");
    expect(PurchaseHistoryCalculator.toDollar(10.1)).toBe("$10.10");
    expect(PurchaseHistoryCalculator.toDollar(10.000000001)).toBe("$10.00");
    expect(PurchaseHistoryCalculator.toDollar(-10.9777777772)).toBe("-$10.98");
  });
});
