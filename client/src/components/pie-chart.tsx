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
  className?: string;
};

export default function Pie_Chart({ labels, datasets,options, className }: ChartProps) {
  return (
    <div className="flex items-center justify-center text-white">
      <Doughnut
        data={{
          labels,
          datasets: datasets.map((dataset) => ({
            data: dataset.data,
            backgroundColor: dataset.backgroundColor || ["#4CAF50", "#FFC107", "#F44336"],
          })),
        }}
        options={options}
        className={className}
        
      />
    </div>
  );
}
