export const sentimentFooter = (score: number, meter: string): string => {
  let trend = "";
  if (meter == "hype") {
    trend = "social media";
  } else if (meter == "impact") {
    trend = "news";
  }
  // Converts sentiment score into a readable sentence based on the meter type
  if (score >= 0 && score <= 15) {
    return `Strongly Negative sentiment in ${trend} around the stock`;
  } else if (score > 15 && score <= 30) {
    return `Negative sentiment in ${trend} around the stock`;
  } else if (score > 30 && score <= 45) {
    return `Slightly Negative sentiment in ${trend} around the stock`;
  } else if (score > 45 && score <= 55) {
    return `Neutral sentiment in ${trend} around the stock`;
  } else if (score > 55 && score <= 70) {
    return `Slightly Positive sentiment in ${trend} around the stock`;
  } else if (score > 70 && score <= 85) {
    return `Positive sentiment in ${trend} around the stock`;
  } else if (score > 85 && score <= 100) {
    return `Strongly Positive sentiment in ${trend} around the stock`;
  } else {
    return "N/A";
  }
};
