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
