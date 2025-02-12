import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function Documentation() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-transparent text-gray-900 dark:text-white px-6 py-10">
      <Card className="max-w-4xl w-full shadow-lg bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-2xl p-6">
        <CardHeader className="text-center">
          <CardTitle className="text-4xl font-bold tracking-tight text-gray-900 dark:text-white">
            Documentation
          </CardTitle>
        </CardHeader>
        <CardContent className="text-lg text-gray-700 dark:text-gray-300">
          <div className="flex flex-col items-center gap-4">
            <a
              className="w-full text-lg font-semibold px-4 py-2 text-gray-900 dark:text-white bg-transparent border border-gray-300 dark:border-gray-700 rounded-lg text-center hover:bg-gray-200 dark:hover:bg-gray-700"
              href="/documentation/introduction"
            >
              Introduction
            </a>
            <a
              className="w-full text-lg font-semibold px-4 py-2 text-gray-900 dark:text-white bg-transparent border border-gray-300 dark:border-gray-700 rounded-lg text-center hover:bg-gray-300 dark:hover:bg-gray-600"
              href="/documentation/tutorials"
            >
              Tutorials
            </a>
            <a
              className="w-full text-lg font-semibold px-4 py-2 text-gray-900 dark:text-white bg-transparent border border-gray-300 dark:border-gray-700 rounded-lg text-center hover:bg-gray-400 dark:hover:bg-gray-500"
              href="/documentation/faq"
            >
              FAQ
            </a>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}