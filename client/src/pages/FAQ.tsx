import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

export default function FAQ() {
  const faqs = [
    {
      question: "What is MarketPulse?",
      answer:
        "MarketPulse is an AI-powered financial market intelligence platform that delivers real-time insights, data-driven analysis, and trend forecasting to empower investors, traders, and financial analysts.",
    },
    {
      question: "How does it predict prices?",
      answer:
        "MarketPulse leverages advanced AI models to analyze historical data, market trends, and social sentiment to predict future prices and provide real-time insights.",
    },
    {
      question: "Can it help with emotional trading?",
      answer:
        "Yes, MarketPulse helps investors make data-driven decisions by providing real-time insights and removing emotions from trading, leading to more consistent and profitable outcomes.",
    },
    {
      question: "Is it beginner-friendly?",
      answer:
        "Absolutely! MarketPulse is designed to help beginner investors overcome emotional trading and time constraints by providing easy-to-understand insights and predictions.",
    },
    {
      question: "What markets does MarketPulse cover?",
      answer:
        "MarketPulse covers the top 5 automotive stocks which are Tesla, Toyota, GM, Ford and Stellantis.",
    },
    {
      question: "How often is the data updated?",
      answer:
        "The data on MarketPulse is updated in real-time to provide the most accurate and up-to-date information.",
    },
    {
      question: "Does Marketpulse buy and sell stocks for me?",
      answer:
        "No, MarketPulse only gives suggestions based on our AI model and sentiment analysis.",
    },
    {
      question: "Is there a mobile app available?",
      answer:
        "No, MarketPulse is currently only a web application.",
    },
    {
      question: "Is there customer support available?",
      answer:
        "Yes, MarketPulse offers 24/7 customer support to assist you with any questions or issues through email.",
    },
  ];
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
            <Accordion type="single" collapsible className="w-full">
              {faqs.map((faq, index) => (
                <AccordionItem key={index} value={`item-${index}`}>
                  <AccordionTrigger>{faq.question}</AccordionTrigger>
                  <AccordionContent>{faq.answer}</AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}