import { type PurchaseHistoryDatapoint } from "@/types/global_state";

export class ForecastModelCalculator {
  private actual: number[];
  private predicted: number[];

  constructor(actual: number[], predicted: number[]) {
    if (actual.length !== predicted.length) {
      throw new Error("actual and predicted arrays must have the same length.");
    }
    this.actual = actual;
    this.predicted = predicted;
  }

  public meanAbsoluteError(): number {
    const totalError = this.actual.reduce((sum, actualValue, index) => {
      return sum + Math.abs(actualValue - this.predicted[index]);
    }, 0);
    return totalError / this.actual.length;
  }

  public meanSquaredError(): number {
    const totalError = this.actual.reduce((sum, actualValue, index) => {
      return sum + Math.pow(actualValue - this.predicted[index], 2);
    }, 0);
    return totalError / this.actual.length;
  }

  public rootMeanSquaredError(): number {
    return Math.sqrt(this.meanSquaredError());
  }

  public rSquared(): number {
    const meanActual =
      this.actual.reduce((sum, value) => sum + value, 0) / this.actual.length;
    const totalSumOfSquares = this.actual.reduce(
      (sum, value) => sum + Math.pow(value - meanActual, 2),
      0
    );
    const residualSumOfSquares = this.actual.reduce(
      (sum, actualValue, index) => {
        return sum + Math.pow(actualValue - this.predicted[index], 2);
      },
      0
    );
    return 1 - residualSumOfSquares / totalSumOfSquares;
  }

  public meanAbsolutePercentageError(): number {
    const totalError = this.actual.reduce((sum, actualValue, index) => {
      return (
        sum + Math.abs((actualValue - this.predicted[index]) / actualValue)
      );
    }, 0);
    return (totalError / this.actual.length) * 100;
  }

  private standardDeviation(values: number[]): number {
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance =
      values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) /
      values.length;
    return Math.sqrt(variance);
  }

  /**
   * @description calculates the accuracy of the predictions
   * @returns {number} - the accuracy of the predictions as a percentage
   * @example
   * const calculator = new ForecastModelCalculator([100, 200], [101, 199]);
   * const accuracy = calculator.accuracy()
   * console.log(accuracy); // 0.75 ?
   */
  public accuracy(): number {
    // Calculate the range of actual values (max - min)
    const minActual = Math.min(...this.actual);
    const maxActual = Math.max(...this.actual);
    const range = maxActual - minActual;
    // TODO: this currently is pretty dishonest. maybe use stdev instead?
    const threshold = range * 0.25;

    let correctPredictions = 0;
    for (let i = 0; i < this.actual.length; i++) {
      if (Math.abs(this.actual[i] - this.predicted[i]) <= threshold) {
        correctPredictions++;
      }
    }
    return correctPredictions / this.actual.length;
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

    for (const { amount_purchased, price_purchased } of this.purchases) {
      const value = amount_purchased * price_purchased;

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
      }
    }
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
    return this.totalShares === 0 ? 0 : this.totalOwned / this.totalShares;
  }

  getTotalValue(currentPrice: number): number {
    return this.totalShares * currentPrice;
  }

  getTotalProfit(currentPrice: number): number {
    return this.getTotalValue(currentPrice) - this.totalOwned;
  }
}
