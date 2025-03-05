import { Label, PolarRadiusAxis, RadialBar, RadialBarChart } from "recharts";

import { Card, CardContent } from "@/components/ui/card";
import { ChartConfig, ChartContainer } from "@/components/ui/chart";

function chooseEmoji(score: number): string {
  if (score <= -4) return "Mostly negative";
  if (score < 0) return "Negative";
  if (score === 0) return "Neutral";
  if (score > 0 && score <= 1) return "Positive";
  if (score > 1 && score <= 4) return "Mostly positive";
  return "Positive";
}
const chartConfig = {
  negative: {
    label: "Negative",
    color: "hsl(var(--chart-1))",
  },
  positive: {
    label: "Positive",
    color: "hsl(var(--chart-2))",
  },
} satisfies ChartConfig;

interface RadialChartProps {
  score: number;
}
export default function RadialChart({ score }: RadialChartProps) {
  let chartData;
  const emotion = chooseEmoji(score);
  if (score > 0) {
    const pos = 6 + score;
    const neg = 12 - pos;
    chartData = [{ positive: pos, negative: neg }];
  } else if (score < 0) {
    const neg = 6 - score;
    const pos = 12 - neg;
    chartData = [{ positive: pos, negative: neg }];
  } else {
    chartData = [{ positive: 0, negative: 0 }];
  }

  return (
    <Card className="flex flex-col w-full h-52">
      <CardContent className="flex flex-1 items-center pt-10 w-full">
        <ChartContainer
          config={chartConfig}
          className="mx-auto aspect-square w-full max-w-[250px]"
        >
          <RadialBarChart
            data={chartData}
            endAngle={180}
            innerRadius={100}
            outerRadius={180}
          >
            <PolarRadiusAxis tick={false} tickLine={false} axisLine={false}>
              <Label
                content={({ viewBox }) => {
                  if (viewBox && "cx" in viewBox && "cy" in viewBox) {
                    return (
                      <text x={viewBox.cx} y={viewBox.cy} textAnchor="middle">
                        <tspan
                          x={viewBox.cx}
                          y={(viewBox.cy || 0) - 16}
                          className="fill-foreground text-2xl font-bold"
                        >
                          {(Number(score / 6) * 100).toFixed(0)}%
                        </tspan>
                        <tspan
                          x={viewBox.cx}
                          y={viewBox.cy || 0}
                          className="fill-muted-foreground"
                        >
                          {emotion}
                        </tspan>
                        <tspan
                          x={viewBox.cx}
                          y={(viewBox.cy || 0) + 16}
                          className="fill-muted-foreground"
                        >
                          sentiment
                        </tspan>
                      </text>
                    );
                  }
                }}
              />
            </PolarRadiusAxis>
            <RadialBar
              dataKey="negative"
              stackId="a"
              cornerRadius={5}
              fill="#F93827"
              className="stroke-transparent stroke-2 hover:stroke-[#F93827] hover:stroke-[0.3rem]"
              style={{ transition: "stroke 0.3s" }}
            />
            <RadialBar
              dataKey="positive"
              fill="#16C47F"
              stackId="a"
              cornerRadius={5}
              className="stroke-transparent stroke-2 hover:stroke-[#16C47F] hover:stroke-[0.3rem]"
              style={{ transition: "stroke 0.3s" }}
            />
          </RadialBarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
