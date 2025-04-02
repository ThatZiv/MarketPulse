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
  private totalOwned: number = 0;
  private totalSold: number = 0;
  private profit: number = 0;

  /**
   * @param purchases - array of purchase history datapoints
   */
  constructor(purchases: PurchaseHistoryDatapoint[]) {
    this.setPurchases(purchases);
  }

  public setPurchases(purchases: PurchaseHistoryDatapoint[]) {
    this.purchases = purchases.sort(
      (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
    );
    this.calculateTotals();
  }

  /**
   * calculates all totals
   */
  private calculateTotals() {
    this.totalShares = 0;
    this.totalBought = 0;
    this.totalOwned = this.totalOwned = 0;
    this.totalSold = 0;
    this.profit = 0;

    //let curr = 0;
    for (const { amount_purchased, price_purchased } of this.purchases) {
      const value = amount_purchased * price_purchased;
      //curr += value;
      this.totalShares += amount_purchased;
      if (amount_purchased > 0) {
        this.totalBought += value;
        this.totalOwned += value;
      } else {
        this.totalSold += Math.abs(value);
        this.profit =
          (this.totalOwned / (this.totalShares - amount_purchased) -
            price_purchased) *
          amount_purchased;
        // the total value of owned shares is reduced by the rolling average * the number of shares sold
        this.totalOwned =
          this.totalOwned +
          (this.totalOwned / (this.totalShares - amount_purchased)) *
            amount_purchased;
        console.log(this.totalOwned);
        //this.profit = curr * -1;
        //curr = 0;
        //continue;
      }
    }

    // this.profit = this.totalSold === 0 ? 0 : this.totalSold - this.totalBought;
  }

  /**
   * checks if the purchase history is valid at all points in time
   * @description check if selling more shares than owned
   * @returns {null|string} returns the date of the first invalid history entry, or null if all history is valid
   * */
  isInvalidHistory(): null | string {
    let currentShares = 0;
    for (const purchase of this.purchases) {
      currentShares += purchase.amount_purchased ?? 0;
      if (currentShares < 0) {
        return purchase.date;
      }
    }
    return null;
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

  getAveragePrice(): number {
    return this.totalShares === 0 ? 0 : this.totalBought / this.totalShares;
  }

  getTotalValue(currentPrice: number): number {
    return this.totalShares * currentPrice;
  }

  getTotalProfit(currentPrice: number): number {
    return this.getTotalValue(currentPrice) - this.totalOwned;
  }
}
