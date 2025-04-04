import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Link } from "react-router";

export default function Tutorials() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-transparent text-gray-900 dark:text-white px-6 py-10">
      <Card className="max-w-4xl w-full shadow-lg bg-white dark:bg-black border border-gray-300 dark:border-gray-700 rounded-2xl p-6">
        <CardHeader className="text-center mb-8">
          <CardTitle className="text-4xl font-bold tracking-tight text-gray-900 dark:text-white">
            How to Use MarketPulse
          </CardTitle>
        </CardHeader>
        <CardContent className="text-lg text-gray-700 dark:text-gray-300">
          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="item-1">
              <AccordionTrigger className="text-xl font-semibold text-gray-900 dark:text-white">
                1. Getting Started
              </AccordionTrigger>
              <AccordionContent className="text-gray-700 dark:text-gray-300">
                <div className="flex flex-col items-start gap-8">
                  {[
                    {
                      fallback: "1",
                      title: "18 and Above",
                      description: "Users must be 18 year old or above.",
                    },
                    {
                      fallback: "2",
                      title: "Basic Stock Market Knowledge",
                      description:
                        "Users should have a basic understanding of stock market concepts.",
                    },
                    {
                      fallback: "3",
                      title: "English Proficiency",
                      description:
                        "MarketPulse is currently available to only English-Speaking Users.",
                    },
                    {
                      fallback: "4",
                      title: "Email Address",
                      description:
                        "Users must have a valid email address to create an account.",
                    },
                  ].map((step, index) => (
                    <div key={index} className="flex items-center gap-4 w-full">
                      <Avatar className="w-24 h-24 bg-white dark:bg-black border border-gray-300 dark:border-gray-700">
                        <AvatarFallback>{step.fallback}</AvatarFallback>
                      </Avatar>
                      <div className="flex-1">
                        <h2 className="text-2xl font-semibold text-gray-900 dark:text-white text-left">
                          {step.title}
                        </h2>
                        <p className="mt-2 text-base text-gray-900 dark:text-gray-300 text-left">
                          {step.description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-2">
              <AccordionTrigger className="text-xl font-semibold text-gray-900 dark:text-white">
                2. Adding Stocks to your portfolio
              </AccordionTrigger>
              <AccordionContent className="text-gray-700 dark:text-gray-300">
                <p className="text-xl">
                  This tutorial will guide you through the steps to add stocks
                  to your portfolio.
                </p>
                <div className="flex flex-col items-start gap-8">
                  {[
                    {
                      fallback: "1",
                      title: "Navigate to the Dashboard",
                      description:
                        "Navigate to the dashboard tab to start your process of adding a new stock to your portfolio.",
                    },
                    {
                      fallback: "2",
                      title: "Add a stock",
                      description:
                        "Click the button with the plus on it to add a new stock.",
                    },
                    {
                      fallback: "3",
                      title: "Fill out the form",
                      description:
                        "Fill out all the required sections of the form which includes stock name, any stock investment history, and desired investitment amount.",
                    },
                    {
                      fallback: "4",
                      title: "Click Submit",
                      description:
                        "Click the submit button to add the stock to your portfolio.",
                    },
                    {
                      fallback: "5",
                      title: "Error Resolution",
                      description: (
                        <>
                          If the stock fails to add to your portfolio, contact
                          the{" "}
                          <Link
                            to="/support"
                            className="hover:font-bold underline active:bg-secondary"
                          >
                            support team
                          </Link>{" "}
                          with the error message.
                        </>
                      ),
                    },
                  ].map((step, index) => (
                    <div key={index} className="flex items-center gap-4 w-full">
                      <Avatar className="w-24 h-24 bg-white dark:bg-black border border-gray-300 dark:border-gray-700">
                        <AvatarFallback>{step.fallback}</AvatarFallback>
                      </Avatar>
                      <div className="flex-1">
                        <h2 className="text-2xl font-semibold text-gray-900 dark:text-white text-left">
                          {step.title}
                        </h2>
                        <p className="mt-2 text-base text-gray-900 dark:text-gray-300 text-left">
                          {step.description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-3">
              <AccordionTrigger className="text-xl font-semibold text-gray-900 dark:text-white">
                3. Understanding Stock Page
              </AccordionTrigger>
              <AccordionContent className="text-gray-700 dark:text-gray-300">
                <p className="text-xl">
                  By the end of this tutorial, you will be able to navigate a
                  stock page and understand key information such as price,
                  market trends, charts, financials, and news.
                </p>
                <div className="flex flex-col items-start gap-8">
                  {[
                    {
                      fallback: "1",
                      title: "Navigate to the Stock Estimation Page",
                      description:
                        "Use the Dashboard or Sidebar to access the Stock Estimation Page for your selected stock.",
                    },
                    {
                      fallback: "2",
                      title: "Overview",
                      description:
                        "The page displays your stock holdings (Stocks) and the current price of the stock (Current Price).",
                    },
                    {
                      fallback: "3",
                      title: "Historical Data and Predictions",
                      description:
                        "A graph below shows Historical Stock Prices along with past predictions made by MarketPulse.",
                    },
                    {
                      fallback: "4",
                      title: "Future Predictions",
                      description:
                        "A line chart displays future stock predictions from five different models, as well as the average prediction.",
                    },
                    {
                      fallback: "5",
                      title: "Predictions Table",
                      description:
                        "Future predictions are also shown in a tabular format with filter and sort options.",
                    },
                    {
                      fallback: "6",
                      title: "Buy/Sell Predictions",
                      description:
                        "The Buy/Sell Predictions section estimates potential buy or sell decisions based on MarketPulse's forecasts.",
                    },
                    {
                      fallback: "6",
                      title: "Hype Meter & Impact Factor",
                      description:
                        "The Hype Meter and Impact Factor provide an assessment of social media and news sentiment regarding the stock.",
                    },
                    {
                      fallback: "7",
                      title: "Purchase History",
                      description:
                        "A graph displays your stock purchase history, if available, along with a Portfolio Summary.",
                    },
                  ].map((step, index) => (
                    <div key={index} className="flex items-center gap-4 w-full">
                      <Avatar className="w-24 h-24 bg-white dark:bg-black border border-gray-300 dark:border-gray-700">
                        <AvatarFallback>{step.fallback}</AvatarFallback>
                      </Avatar>
                      <div className="flex-1">
                        <h2 className="text-2xl font-semibold text-gray-900 dark:text-white text-left">
                          {step.title}
                        </h2>
                        <p className="mt-2 text-base text-gray-900 dark:text-gray-300 text-left">
                          {step.description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-4">
              <AccordionTrigger className="text-xl font-semibold text-gray-900 dark:text-white">
                4. Historical Stock Prices and Predictions Chart
              </AccordionTrigger>
              <AccordionContent className="text-gray-700 dark:text-gray-300">
                <p className="text-xl">
                  This tutorial will give you detailed information about the
                  Historical Stock Prices and Predictions Chart.
                </p>
                <div className="flex flex-col items-start gap-8">
                  {[
                    {
                      fallback: "1",
                      title: "Overview",
                      description:
                        "Historical Stock Prices and Predictions Chart is within Stock Estimation Page of a Stock. It contains graph with both historical prices and past predictions made by MarketPulse.",
                    },
                    {
                      fallback: "2",
                      title: "Dropdown",
                      description: `A dropdown box is located in the top right corner, allowing users to switch between "Last 14 days" and "Last 7 days" options.`,
                    },
                    {
                      fallback: "3",
                      title: "Hover on chart",
                      description:
                        "Users can hover over the chart to view both historical and predicted price at a specific date.",
                    },
                  ].map((step, index) => (
                    <div key={index} className="flex items-center gap-4 w-full">
                      <Avatar className="w-24 h-24 bg-white dark:bg-black border border-gray-300 dark:border-gray-700">
                        <AvatarFallback>{step.fallback}</AvatarFallback>
                      </Avatar>
                      <div className="flex-1">
                        <h2 className="text-2xl font-semibold text-gray-900 dark:text-white text-left">
                          {step.title}
                        </h2>
                        <p className="mt-2 text-base text-gray-900 dark:text-gray-300 text-left">
                          {step.description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-5">
              <AccordionTrigger className="text-xl font-semibold text-gray-900 dark:text-white">
                5. Future Price Predictions Chart
              </AccordionTrigger>
              <AccordionContent className="text-gray-700 dark:text-gray-300">
                <p className="text-xl">
                  This tutorial will help you navigate the Future Price
                  Predictions Chart.
                </p>
                <div className="flex flex-col items-start gap-8">
                  {[
                    {
                      fallback: "1",
                      title: "Overview",
                      description:
                        "Future Predictions contains stock price predictions from 5 different models along with average predictions of all models. All of them are shown in different colors.",
                    },
                    {
                      fallback: "2",
                      title: "Predictions Range",
                      description:
                        "The chart displays predictions for the next 1 or 7 days, with the x-axis representing the date and the y-axis representing the predicted price. This range can be set using the dropdown from the table underneath the graph.",
                    },
                    {
                      fallback: "3",
                      title: "Hover on chart",
                      description:
                        "Users can hover over the chart to view price predictions from all models along with average price predictions.",
                    },
                    {
                      fallback: "4",
                      title: "Actions from Prediction Table Underneath",
                      description:
                        "Users can sort and filter the models in the table based on their preferences. The table contains the same information as the chart, but in a tabular format.",
                    },
                    {
                      fallback: "5",
                      title: "Model Information",
                      description:
                        "More information about the models can be found at Model Information Section in the Tutorials",
                    },
                  ].map((step, index) => (
                    <div key={index} className="flex items-center gap-4 w-full">
                      <Avatar className="w-24 h-24 bg-white dark:bg-black border border-gray-300 dark:border-gray-700">
                        <AvatarFallback>{step.fallback}</AvatarFallback>
                      </Avatar>
                      <div className="flex-1">
                        <h2 className="text-2xl font-semibold text-gray-900 dark:text-white text-left">
                          {step.title}
                        </h2>
                        <p className="mt-2 text-base text-gray-900 dark:text-gray-300 text-left">
                          {step.description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="item-6">
              <AccordionTrigger
                id="model-info"
                className="text-xl font-semibold text-gray-900 dark:text-white"
              >
                6. Model Information
              </AccordionTrigger>
              <AccordionContent className="text-gray-700 dark:text-gray-300">
                <p className="text-xl">
                  This section provides more information about the stock
                  prediction models used in MarketPulse
                </p>
                <div className="flex flex-col items-start gap-8">
                  {[
                    {
                      fallback: "1",
                      title: "CNN-LSTM",
                      description:
                        "CNN-LSTM (Convolutional Neural Network - Long Short-Term Memory) is a hybrid deep learning model that combines the strengths of Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. The CNN component is used for extracting spatial features from data (such as images or sequences), while the LSTM component captures temporal dependencies or sequential patterns in the data.",
                    },
                    {
                      fallback: "2",
                      title: "Attention-LSTM",
                      description:
                        "Attention-LSTM (Long Short-Term Memory with Attention) is a neural network model that combines the capabilities of LSTM and an attention mechanism. The LSTM component is used to capture long-term dependencies in sequential data, while the attention mechanism allows the model to focus on the most relevant parts of the sequence at each step, improving the model's performance on tasks where certain inputs are more important than others",
                    },
                    {
                      fallback: "3",
                      title: "ARIMA/SARIMA",
                      description:
                        "ARIMA (AutoRegressive Integrated Moving Average) is a statistical model used for forecasting time series data by combining three components: autoregression (AR), differencing (I), and moving average (MA). The AR part models the relationship between the current value and previous observations, the I part makes the data stationary by removing trends, and the MA part models errors from past predictions. ARIMA is commonly represented as ARIMA(p, d, q), where p, d, and q are parameters for each component, respectively.",
                    },
                    {
                      fallback: "4",
                      title: "Transformer",
                      description:
                        "a Transformer is a deep learning model that uses self-attention mechanisms to capture dependencies and patterns in sequential data. Unlike traditional methods, which process time steps one by one, the Transformer model analyzes all time steps in parallel, enabling it to learn long-term dependencies and relationships more effectively.",
                    },
                    {
                      fallback: "5",
                      title: "XGBoost",
                      description:
                        "XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library. It builds an ensemble of decision trees in a sequential manner, where each new tree corrects the errors made by the previous ones, aiming to minimize the overall prediction error and is the leading machine learning library for regression, classification, and ranking problems.",
                    },
                  ].map((step, index) => (
                    <div key={index} className="flex items-center gap-4 w-full">
                      <Avatar className="w-24 h-24 bg-white dark:bg-black border border-gray-300 dark:border-gray-700">
                        <AvatarFallback>{step.fallback}</AvatarFallback>
                      </Avatar>
                      <div className="flex-1">
                        <h2 className="text-2xl font-semibold text-gray-900 dark:text-white text-left">
                          {step.title}
                        </h2>
                        <p className="mt-2 text-base text-gray-900 dark:text-gray-300 text-left">
                          {step.description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </CardContent>
      </Card>
    </div>
  );
}
