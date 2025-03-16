import { type PurchaseHistoryDatapoint } from "@/types/global_state";

export class Calculator {
  static generateHype(...sentiment_data: number[]): number {
    return (
      sentiment_data.reduce((acc, curr) => acc + curr, 0) /
      sentiment_data.length
    );
  }

  static generateImpact(...news_data: number[]): number {
    return news_data.reduce((acc, curr) => acc + curr, 0) / news_data.length;
  }
}

/**
 * class for calculating purchase history based on data from the database
 */
export class PurchaseHistoryCalculator {
  private purchases: PurchaseHistoryDatapoint[] = [];
  private totalShares: number = 0;
  private totalBought: number = 0;
  private totalSold: number = 0;
  private profit: number = 0;

  /**
   * @param purchases - array of purchase history datapoints
   */
  constructor(purchases: PurchaseHistoryDatapoint[]) {
    this.purchases = purchases.sort(
      (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
    );
    this.calculateTotals();
  }

  /**
   * calculates all totals
   */
  private calculateTotals() {
    this.totalShares = this.purchases.reduce(
      (acc, purchase) => acc + (purchase.amount_purchased ?? 0),
      0
    );

    this.totalBought = this.purchases
      .filter(
        (purchase) =>
          purchase.amount_purchased !== null && purchase?.amount_purchased > 0
      )
      .reduce(
        (acc, purchase) =>
          acc +
          (purchase.price_purchased ?? 0) * (purchase.amount_purchased ?? 0),
        0
      );

    this.totalSold = this.purchases
      .filter(
        (purchase) =>
          purchase.amount_purchased !== null && purchase?.amount_purchased < 0
      )
      .reduce(
        (acc, purchase) =>
          acc +
          (purchase.price_purchased ?? 0) *
            (purchase.amount_purchased ?? 0) *
            -1,
        0
      );
    let curr = 0;
    if (this.totalSold > 0) {
      for (const purchase of this.purchases) {
        curr +=
          (purchase.price_purchased ?? 0) * (purchase.amount_purchased ?? 0);
        if (purchase.amount_purchased && purchase.amount_purchased < 0) {
          // if sell
          this.profit = curr * -1;
          curr = 0;
          continue;
        }
      }
    }
  }

  getTotalShares(): number {
    return this.totalShares;
  }

  /**
   * @returns {number} total $ bought
   *  */
  getTotalBought(): number {
    return this.totalBought;
  }

  /**
   * @returns {number} total $ sold
   *  */
  getTotalSold(): number {
    return this.totalSold;
  }

  getProfit(): number {
    return this.profit;
  }
  /**
   * converts amount to dollar formatted string with proper decimals and negative sign
   * @param amount - amount to convert to dollar string
   * @returns {string} - dollar formatted string
   */
  static toDollar(amount: number): string {
    return `${amount < 0 ? "-$" : "$"}${Math.abs(amount).toLocaleString(
      undefined,
      {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }
    )}`;
  }

  getTotalSpent(): number {
    return this.totalBought + this.profit;
  }

  getAveragePrice(): number {
    return this.totalBought / this.totalShares;
  }

  getTotalValue(currentPrice: number): number {
    return this.totalShares * currentPrice;
  }

  getTotalProfit(currentPrice: number): number {
    return this.getTotalValue(currentPrice) - this.getTotalSpent();
  }
}
