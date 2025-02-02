import { useSupabase } from "@/database/SupabaseProvider";
import Stock_Chart from "@/components/stock_chart_demo";
import Pie_Chart from "@/components/pie-chart";
import { Progress } from "@/components/ui/progress"
import { IoMdInformationCircleOutline, IoMdInformationCircle } from "react-icons/io";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card"
import { Link, useNavigate, useParams } from "react-router-dom";
import { useEffect } from "react";
import { MdEdit } from "react-icons/md";

const hype_meter_labels = ["Positive", "Negative", "Neutral"];
const hype_meter_dataset = [
  {
    data: [50, 20, 30],
    backgroundColor: ["#4CAF50", "#FFC107", "#F44336"],
  },
];
const hype_meter_options = {
  responsive: true,
};
const options = {
  responsive: true,
  animation: {
    animateScale: true,
  },
  maintainAspectRatio: false,
  cutout: "50%", // Adjust for thickness (50% for a classic doughnut)
  rotation: -90, // Start from top
  circumference: 180, // Only show half of the circle
  plugins: {
    legend: {
      display: true,
    },
  },
};

const buyScore = 44; // Future: can change this dynamically
const sellScore = 100 - buyScore;

const availableStocks = [
  {"TSLA": "Tesla"},
  {"F":"Ford"},
  {"GM":"General Motors"},
  {"TM":"Toyota Motor Corporation"},
  {"RIVN": "Rivian Automotive"},
]

export default function Stocks() {
  const { user } = useSupabase();
  const { ticker }: { ticker?: string } = useParams();
  const navigate = useNavigate();
  const stock = availableStocks.find(stock => stock[ticker as keyof typeof stock]);
  useEffect(() => {
    if (!stock) {
      // Redirect
      navigate("/")
    }
  });
  return (
    <div className="lg:p-4 md:w-9/12 w-8/12">
      <h1 className="font-semibold text-3xl pb-6">{stock ? stock[ticker as keyof typeof stock] || "Undefined" : "Stock not found"}</h1>
      <div className="border border-black p-4 bg-secondary rounded-lg">
        <div className="relative">
          <Link to="/stocks">
          <MdEdit className="absolute right-0 top-1/2 transform -translate-y-1/2 transition-transform duration-300 hover:scale-125"/>
          </Link>
        </div>
        
        <h2 className="font-semibold md:text-md text-xs">Hey {user?.email ?? "Guest"},</h2>
        <h3 className="md:text-md text-xs">Current Stock Rate: $ 10.12</h3>
        <div className="flex flex-row justify-center lg:gap-64 md:gap-32 gap:5 mt-4">
          <div className="flex flex-col">
            <h3 className="lg:text-lg text-md">Number of Stocks Invested:</h3>
            <p className="lg:text-4xl md:text-3xl text-2xl">10</p>
          </div>
          <div className="flex flex-col">
            <h3 className="lg:text-lg text-md">Current Stock Earnings:</h3>
            <p className="lg:text-4xl md:text-3xl text-2xl">$101.12</p>
          </div>
        </div>
      </div>
      <div className="border border-black p-6 bg-secondary rounded-lg mt-4">
        <h2 className="font-semibold text-xl pb-2">Sell: {sellScore}% Buy: {buyScore}%</h2>
        <Progress value={sellScore} />
      </div>
      <div className="flex flex-col gap-4 mt-4 max-w-screen h-2/3">
        <div className="flex flex-col items-center justify-between border border-black">
          <div className="flex flex-row gap-2 pt-2">
            <h3 className="text-center font-semibold">Hype Meter</h3>
            <HoverCard>
              <HoverCardTrigger>
                <IoMdInformationCircleOutline className="mt-[0.25rem] dark:hidden" />
                <IoMdInformationCircle className="mt-[0.25rem] invisible dark:visible dark:block" />
              </HoverCardTrigger>
              <HoverCardContent>
                Hype meter is used for getting sentiment analysis from social media to predict the stock market.
              </HoverCardContent>
            </HoverCard>
          </div>
          <Pie_Chart labels={hype_meter_labels} datasets={hype_meter_dataset} options={hype_meter_options}/>
        </div>
        <div className="flex flex-col lg:flex-row justify-between gap-4 mt-4 max-w-screen">

          <div className="flex flex-col items-center justify-between border border-black w-full h=6/5">
            <div className="flex flex-row gap-2 pt-2">
              <h3 className="text-center font-semibold">Disruption Score</h3>
              <HoverCard>
                <HoverCardTrigger>
                  <IoMdInformationCircleOutline className="mt-[0.25rem] dark:hidden" />
                  <IoMdInformationCircle className="mt-[0.25rem] invisible dark:visible dark:block" />
                </HoverCardTrigger>
                <HoverCardContent>
                  A "Disruption Score" that evaluates potential impacts on stock prices due to
                  supply chain delays or shifts.
                </HoverCardContent>
              </HoverCard>
            </div>
            <Pie_Chart labels={hype_meter_labels} datasets={hype_meter_dataset} options={options} />
          </div>
          <div className="flex flex-col items-center justify-between border border-black w-full h-auto">
            <div className="flex flex-row gap-2 pt-2">
              <h3 className="text-center font-semibold">Impact Factor</h3>
              <HoverCard>
                <HoverCardTrigger>
                  <IoMdInformationCircleOutline className="mt-[0.25rem] dark:hidden" />
                  <IoMdInformationCircle className="mt-[0.25rem] invisible dark:visible dark:block" />
                </HoverCardTrigger>
                <HoverCardContent>
                An "Impact Factor" that scores how major events (e.g., elections, natural
                  disasters, regulations) may influence stock performance.
                </HoverCardContent>
              </HoverCard>
            </div>
            <Pie_Chart labels={hype_meter_labels} datasets={hype_meter_dataset} options={options} />
          </div>
        </div>
      </div>
    </div>
  );
}
