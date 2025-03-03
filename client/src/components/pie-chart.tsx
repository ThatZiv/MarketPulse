"use client";

import { Pie, PieChart, Sector } from "recharts";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { useState } from "react";

const chartData = [
  { action: "buy", suggest: 50, fill: "var(--color-buy)" },
  { action: "sell", suggest: 20, fill: "var(--color-sell)" },
  { action: "hold", suggest: 30, fill: "var(--color-hold)" },
];

const chartConfig = {
  buy: {
    label: "Buy",
    color: "hsl(var(--chart-1))",
  },
  sell: {
    label: "Sell",
    color: "hsl(var(--chart-2))",
  },
  hold: {
    label: "Hold",
    color: "hsl(var(--chart-3))",
  },
};

export function Pie_Chart() {
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  const renderActiveShape = (props: any) => {
    const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill } = props;

    return (
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius + 10}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
      />
    );
  };

  return (
    <Card className="flex flex-col w-full h-full">
      <CardHeader className="items-center pb-0">
        <CardTitle className="text-xl">Buy/Sell/Hold Suggestions:</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 pb-0">
        <ChartContainer
          config={chartConfig}
          className="mx-auto aspect-square max-h-[250px]"
        >
          <PieChart>
            <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
            <Pie
              data={chartData}
              dataKey="suggest"
              nameKey="action"
              innerRadius={60}
              outerRadius={80}
              activeShape={renderActiveShape}
              onMouseEnter={(_, index) => setActiveIndex(index)}
              onMouseLeave={() => setActiveIndex(null)}
              className="stroke-transparent stroke-2 hover:stroke-[0.3rem]"
              style={{ transition: "stroke 1s" }}
            />
          </PieChart>
        </ChartContainer>
      </CardContent>
      <div className="chart-labels flex justify-around mt-4 pb-3">
        <div className="label-item flex items-center space-x-2">
          <div
            className="w-4 h-4"
            style={{ backgroundColor: chartConfig.buy.color }}
          ></div>
          <span style={{ color: chartConfig.buy.color }}>{chartConfig.buy.label}</span>
        </div>
        <div className="label-item flex items-center space-x-2">
          <div
            className="w-4 h-4"
            style={{ backgroundColor: chartConfig.sell.color }}
          ></div>
          <span style={{ color: chartConfig.sell.color }}>{chartConfig.sell.label}</span>
        </div>
        <div className="label-item flex items-center space-x-2">
          <div
            className="w-4 h-4"
            style={{ backgroundColor: chartConfig.hold.color }}
          ></div>
          <span style={{ color: chartConfig.hold.color }}>{chartConfig.hold.label}</span>
        </div>
      </div>

    </Card>
  );
}
