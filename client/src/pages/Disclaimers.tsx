import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function Disclaimer() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-transparent text-gray-900 dark:text-white px-6 py-10">
      <Card className="max-w-4xl w-full shadow-lg bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-2xl p-6">
        <CardHeader className="text-center">
          <CardTitle className="text-4xl font-bold tracking-tight text-gray-900 dark:text-white">
            Disclaimer
          </CardTitle>
        </CardHeader>
        <CardContent className="text-left text-gray-700 dark:text-gray-300 flex flex-wrap gap-4">
          <p className="w-full">
            The information provided by MarketPulse, including AI-generated
            financial recommendations and outputs, is for informational purposes
            only. It is not intended as financial advice, investment advice, or
            a recommendation to buy, sell, or hold any security or asset.
            MarketPulse generates suggestions based on historical data and
            algorithms and does not guarantee any outcomes or future
            performance. <span className="italic">MarketPulse AI</span> relies
            on low-quality large-language models (LLMs) which occasionally
            provides incorrect and/or misleading results.
          </p>
          <p>
            You should conduct your own research and consult with a qualified
            financial advisor before making <strong>any</strong> investment
            decisions. MarketPulse is not liable for any financial losses,
            damages, or other consequences arising from the use of this platform
            or reliance on its outputs. Past performance is not indicative of
            future results.
          </p>
          <p className="font-bold text-sm">
            Investing in financial markets carries inherent risks. Consider your
            risk tolerance before making any trades.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
