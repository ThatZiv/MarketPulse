import React from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Sparkles,
  TrendingUp,
  BarChart3,
  AlertTriangle,
  Clock,
} from "lucide-react";

export default function Introduction() {
  const navigate = useNavigate();

  const handleGetStartedClick = () => {
    navigate("/");
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-transparent text-gray-900 dark:text-white px-6">
      <Card className="max-w-3xl text-center shadow-xl bg-white dark:bg-black border border-gray-300 dark:border-gray-700 rounded-2xl p-8">
        <CardHeader>
          <CardTitle className="text-4xl font-bold tracking-tight text-gray-900 dark:text-white">
            Welcome to{" "}
            <span className="text-blue-500 dark:text-blue-400">
              MarketPulse
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent className="text-lg text-gray-700 dark:text-gray-300">
          <p>
            Investing can be overwhelming, especially for{" "}
            <strong>beginner investors</strong> who often make emotional
            decisions, leading to significant losses. Many lack the time to
            research the market daily, missing key opportunities.
          </p>
          <div className="mt-4 flex flex-col gap-3">
            <div className="flex items-center text-red-500">
              <AlertTriangle className="w-6 h-6 mr-2" />
              <span className="text-md">Emotional Trading Losses</span>
            </div>
            <div className="flex items-center text-yellow-500">
              <Clock className="w-6 h-6 mr-2" />
              <span className="text-md">Lack of Research Time</span>
            </div>
          </div>
          <Separator className="my-6 border-gray-300 dark:border-gray-600" />
          <p>
            <span className="text-blue-500 dark:text-blue-400 font-semibold">
              MarketPulse
            </span>{" "}
            solves this by leveraging{" "}
            <strong>AI-driven market intelligence</strong> to predict future
            prices and analyze social sentiment in real-time. Get instant
            insights, remove emotions from your trades, and make data-driven
            decisions effortlessly.
          </p>
          <Separator className="my-6 border-gray-300 dark:border-gray-600" />
          <div className="grid md:grid-cols-3 gap-6">
            <FeatureCard
              icon={
                <TrendingUp
                  size={32}
                  className="text-blue-500 dark:text-blue-400"
                />
              }
              title="Real-Time Market Trends"
              description="Stay ahead with AI-driven market sentiment analysis and up-to-the-minute trends."
            />
            <FeatureCard
              icon={
                <BarChart3
                  size={32}
                  className="text-green-500 dark:text-green-400"
                />
              }
              title="Advanced Data Analytics"
              description="Leverage powerful data visualization and predictive analytics to make informed decisions."
            />
            <FeatureCard
              icon={
                <Sparkles
                  size={32}
                  className="text-yellow-500 dark:text-yellow-400"
                />
              }
              title="AI-Powered Insights"
              description="Harness the power of AI to uncover hidden opportunities in the financial markets."
            />
          </div>
          <Separator className="my-6 border-gray-300 dark:border-gray-600" />
          <p>
            <strong>MarketPulse</strong> is designed to help beginner investors
            overcome the challenges of emotional trading and lack of time for
            market research. By using advanced AI models, MarketPulse predicts
            future prices and provides real-time sentiment analysis to keep you
            informed about the latest market activities.
          </p>
          <Separator className="my-6 border-gray-300 dark:border-gray-600" />
          <div className="mt-8">
            <Button
              className="px-6 py-3 text-lg font-semibold shadow-md bg-blue-600 hover:bg-blue-700 transition rounded-full"
              onClick={handleGetStartedClick}
            >
              Get Started
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
}

const FeatureCard: React.FC<FeatureCardProps> = ({
  icon,
  title,
  description,
}) => {
  return (
    <div className="flex flex-col items-center text-center p-4 bg-white dark:bg-black rounded-xl shadow-md border border-gray-300 dark:border-gray-600">
      <div className="mb-3">{icon}</div>
      <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
        {title}
      </h3>
      <p className="text-gray-700 dark:text-gray-400 text-sm mt-2">
        {description}
      </p>
    </div>
  );
};
