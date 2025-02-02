import { Doughnut } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement, // Required for Pie & Doughnut charts
  Tooltip,
  Legend,
  ChartDataset,
} from "chart.js";

ChartJS.register(ArcElement, Tooltip, Legend);

type Data = {
  data: number[]; // ✅ Changed from `datasets` to `data`
  backgroundColor?: string[];
};

type ChartProps = {
  labels: string[];
  datasets: Data[]; // ✅ Ensure it's an array of objects, each with `data`
};

export default function Sentiment_Chart({ labels, datasets }: ChartProps) {
  return (
    <div>
      <Doughnut
        data={{
          labels,
          datasets: datasets.map((dataset) => ({
            data: dataset.data,
            backgroundColor: dataset.backgroundColor || ["#4CAF50", "#FFC107", "#F44336"],
          })),
        }}
      />
    </div>
  );
}
