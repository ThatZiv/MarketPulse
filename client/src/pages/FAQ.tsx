import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem } from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";

export default function FAQ() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-transparent text-gray-900 dark:text-white px-6 py-10">
      <Card className="max-w-4xl w-full shadow-lg bg-white dark:bg-black border border-gray-300 dark:border-gray-700 rounded-2xl p-6">
        <CardHeader className="text-center">
          <CardTitle className="text-4xl font-bold tracking-tight text-gray-900 dark:text-white">
            Frequently Asked Questions
          </CardTitle>
        </CardHeader>
        <CardContent className="text-lg text-gray-700 dark:text-gray-300 overflow-y-auto">
          <div className="flex flex-col items-center gap-4">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button className="w-full text-lg font-semibold px-4 py-2 text-gray-900 dark:text-white bg-white dark:bg-transparent border border-gray-300 dark:border-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700">
                  What is MarketPulse?
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-64 p-4 bg-white dark:bg-black border border-gray-300 dark:border-gray-700 rounded-lg">
                <DropdownMenuItem className="leading-relaxed text-gray-900 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600">
                  MarketPulse is an AI-powered financial market intelligence platform that delivers real-time insights, data-driven analysis, and trend forecasting to empower investors, traders, and financial analysts.
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button className="w-full text-lg font-semibold px-4 py-2 text-gray-900 dark:text-white bg-white dark:bg-transparent border border-gray-300 dark:border-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600">
                  How does it predict prices?
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-64 p-4 bg-white dark:bg-black border border-gray-300 dark:border-gray-700 rounded-lg">
                <DropdownMenuItem className="leading-relaxed text-gray-900 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600">
                  MarketPulse leverages advanced AI models to analyze historical data, market trends, and social sentiment to predict future prices and provide real-time insights.
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button className="w-full text-lg font-semibold px-4 py-2 text-gray-900 dark:text-white bg-white dark:bg-transparent border border-gray-300 dark:border-gray-700 rounded-lg hover:bg-gray-400 dark:hover:bg-gray-500">
                  Can it help with emotional trading?
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-64 p-4 bg-white dark:bg-black border border-gray-300 dark:border-gray-700 rounded-lg">
                <DropdownMenuItem className="leading-relaxed text-gray-900 dark:text-gray-300 hover:bg-gray-400 dark:hover:bg-gray-500">
                  Yes, MarketPulse helps investors make data-driven decisions by providing real-time insights and removing emotions from trading, leading to more consistent and profitable outcomes.
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button className="w-full text-lg font-semibold px-4 py-2 text-gray-900 dark:text-white bg-white dark:bg-transparent border border-gray-300 dark:border-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700">
                  Is it beginner-friendly?
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-64 p-4 bg-white dark:bg-black border border-gray-300 dark:border-gray-700 rounded-lg">
                <DropdownMenuItem className="leading-relaxed text-gray-900 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700">
                  Absolutely! MarketPulse is designed to help beginner investors overcome emotional trading and time constraints by providing easy-to-understand insights and predictions.
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button className="w-full text-lg font-semibold px-4 py-2 text-gray-900 dark:text-white bg-white dark:bg-transparent border border-gray-300 dark:border-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600">
                  What markets does MarketPulse cover?
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-64 p-4 bg-white dark:bg-black border border-gray-300 dark:border-gray-700 rounded-lg">
                <DropdownMenuItem className="leading-relaxed text-gray-900 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600">
                  MarketPulse covers the top 5 automotive stocks which are Tesla, Toyota, GM, Ford and Rivian.
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button className="w-full text-lg font-semibold px-4 py-2 text-gray-900 dark:text-white bg-white dark:bg-transparent border border-gray-300 dark:border-gray-700 rounded-lg hover:bg-gray-400 dark:hover:bg-gray-500">
                  How often is the data updated?
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-64 p-4 bg-white dark:bg-black border border-gray-300 dark:border-gray-700 rounded-lg">
                <DropdownMenuItem className="leading-relaxed text-gray-900 dark:text-gray-300 hover:bg-gray-400 dark:hover:bg-gray-500">
                  The data on MarketPulse is updated in real-time to provide the most accurate and up-to-date information.
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button className="w-full text-lg font-semibold px-4 py-2 text-gray-900 dark:text-white bg-white dark:bg-transparent border border-gray-300 dark:border-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700">
                  Does Marketpulse buy and sell stocks for me?
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-64 p-4 bg-white dark:bg-black border border-gray-300 dark:border-gray-700 rounded-lg">
                <DropdownMenuItem className="leading-relaxed text-gray-900 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700">
                  No, MarketPulse only gives suggestions based on our AI model and sentiment analysis.
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button className="w-full text-lg font-semibold px-4 py-2 text-gray-900 dark:text-white bg-white dark:bg-transparent border border-gray-300 dark:border-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600">
                  Is there a mobile app available?
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-64 p-4 bg-white dark:bg-black border border-gray-300 dark:border-gray-700 rounded-lg">
                <DropdownMenuItem className="leading-relaxed text-gray-900 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600">
                  No, MarketPulse is currently only a web application.
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button className="w-full text-lg font-semibold px-4 py-2 text-gray-900 dark:text-white bg-white dark:bg-transparent border border-gray-300 dark:border-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700">
                  Is there customer support available?
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-64 p-4 bg-white dark:bg-black border border-gray-300 dark:border-gray-700 rounded-lg">
                <DropdownMenuItem className="leading-relaxed text-gray-900 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700">
                  Yes, MarketPulse offers 24/7 customer support to assist you with any questions or issues through email.
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}