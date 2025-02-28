import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

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
          <div className="flex flex-col items-start gap-8">
            {[
              {
                fallback: "1",
                title: "Navigating to the Dashboard",
                description: "Navigate to the dashboard tab to start your process of adding a new stock to your portfolio.",
              },
              {
                fallback: "2",
                title: "Add a stock",
                description: "Click the button with the plus on it to add a new stock.",
              },
              {
                fallback: "3",
                title: "Fill out the form",
                description: "Fill out all the required sections of the form and press submit.",
              },
              {
                fallback: "4",
                title: "View analytics",
                description: "Press on your specific stock you want to see the analytics on under the dashboard tab on the sidebar.",
              },
              {
                fallback: "5",
                title: "Read the suggestion",
                description: "View the analytics page where it will mention current social sentiment and the next day's stock price prediction alongside a suggestion on whether you should buy or sell.",
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
                  <p className="mt-2 leading-relaxed text-gray-900 dark:text-gray-300 text-left">
                    {step.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}