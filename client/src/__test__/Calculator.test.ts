import {
  ForecastModelCalculator,
  PurchaseHistoryCalculator,
} from "@/lib/Calculator";
import { PurchaseHistoryDatapoint } from "@/types/global_state";
import "@testing-library/jest-dom";

describe("Forecast Model Calculator", () => {
  const actual = [1, 2, 3, 4, 5];
  const predicted = [1, 2, 3, 5, 4];
  const calc = new ForecastModelCalculator(actual, predicted);
  test("should calculate mean absolute error", () => {
    expect(calc.meanAbsoluteError()).toBe(0.4);
  });
  test("should calculate mean squared error", () => {
    expect(calc.meanSquaredError()).toBe(0.4);
  });
  test("should calculate root mean squared error", () => {
    expect(calc.rootMeanSquaredError()).toBe(0.6324555320336759);
  });
  test("should calculate r squared", () => {
    expect(calc.rSquared()).toBe(0.8);
  });
  test("should calculate accuracy and more", () => {
    expect(calc.accuracy()).toBe(1);
    const actual2 = [1, 2, 3, 4, 5];
    const predicted2 = [1, 2, 3, 2, 15];
    const calc2 = new ForecastModelCalculator(actual2, predicted2);
    expect(calc2.accuracy()).toBe(0.6);
  });
  test("should throw error if arrays are not the same length", () => {
    const actual2 = [1, 2, 3];
    const predicted2 = [1, 2, 3, 4];
    expect(() => {
      new ForecastModelCalculator(actual2, predicted2);
    }).toThrow("actual and predicted arrays must have the same length.");
  });
});

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
