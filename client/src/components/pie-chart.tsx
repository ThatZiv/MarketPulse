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
  data: number[]; 
  backgroundColor?: string[];
};

type ChartProps = {
  labels: string[];
  datasets: Data[];
  options?: object;
};

export default function Pie_Chart({ labels, datasets,options }: ChartProps) {
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
        options={options}
        className="h-full w-full"
        
      />
    </div>
  );
}
