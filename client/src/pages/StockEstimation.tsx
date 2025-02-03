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
import GaugeComponent from 'react-gauge-component';
import React from "react";

const availableStocks = [
  { "TSLA": "Tesla" },
  { "F": "Ford" },
  { "GM": "General Motors" },
  { "TM": "Toyota Motor Corporation" },
  { "RIVN": "Rivian Automotive" },
]
const meters = [
  { "Hype Meter": "Hype Meter analyzes social media sentiment to forecast stock market trends." },
  { "Disruption Score": "Disruption Score measures the potential impact on stock prices from supply chain delays or shifts." },
  { "Impact Factor": "Impact Factor scores how major events like elections, natural disasters, and regulations influence stock performance." },
]
export default function Stocks() {
  const { displayName } = useSupabase();
  const { ticker }: { ticker?: string } = useParams();
  const navigate = useNavigate();
  const stock = availableStocks.find(stock => stock[ticker as keyof typeof stock]);
  useEffect(() => {
    if (!stock) {
      // Redirect
      navigate("/")
    }
  });
  const hype_meter_labels = ["Positive", "Negative", "Neutral"];
  const hype_meter_dataset = [
    {
      data: [50, 20, 30],
      backgroundColor: ["#4CAF50", "#FFC107", "#F44336"],
    },
  ];
  const hype_meter_design = "!lg:w-72 !lg:h-27 w-52 h-52"
  const impact_factor = 10;
  const disruption_score = 40;
  const hype_meter_options = {
    responsive: true,
    animation: {
      animateScale: true,
    },
    maintainAspectRatio: false,
    cutout: "50%", 
    rotation: -90, 
    circumference: 180,
    plugins: {
      legend: {
        display: true,
        labels: {
          color: '#7491b5',
        },
      },
    },
  };
  
  const buyScore = 35// Future: can change this dynamically
  const sellScore = 100 - buyScore;
  
  return (
    <div className="lg:p-4 md:w-10/12 w-full">
      <h1 className="font-semibold text-3xl pb-6">{stock ? stock[ticker as keyof typeof stock] || "Undefined" : "Stock not found"}</h1>
      <div className="border border-black dark:border-white p-4 bg-secondary dark:bg-tertiary/20 rounded-md w-full">
        <div className="relative">
          <Link to="/stocks">
            <MdEdit className="absolute right-0 top-1/2 transform -translate-y-1/2 transition-transform duration-300 hover:scale-125" />
          </Link>
        </div>

        <h2 className="font-semibold md:text-lg text-xs">Hey {displayName},</h2>
        <h3 className="md:text-md text-xs">Current Stock Rate: $ 10.12</h3>
        <div className="flex md:flex-row flex-col justify-center lg:gap-64 md:gap-32 gap:5 mt-4">
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
      <div className="border border-black dark:border-white p-6 bg-secondary dark:bg-tertiary/20 rounded-md mt-4">
        <h2 className="font-semibold text-xl pb-2">Buy: {buyScore}% Sell: {sellScore}%</h2>
        <Progress value={buyScore} />
      </div>
      <div className="flex flex-col md:items-center gap-4 mt-4 w-full">
        <div className="border border-black dark:border-white  bg-secondary rounded-md dark:bg-tertiary/20 md:p-4">
          <div className="flex flex-row justify-center gap-2 pt-2">
            <h3 className="text-center font-semibold text-xl">{Object.keys(meters[0])[0]}</h3>
            <HoverCard>
              <HoverCardTrigger>
                <IoMdInformationCircleOutline className="mt-[0.25rem] dark:hidden" />
                <IoMdInformationCircle className="mt-[0.25rem] invisible dark:visible dark:block" />
              </HoverCardTrigger>
              <HoverCardContent>
              {meters[0]["Hype Meter"]}
              </HoverCardContent>
            </HoverCard>
          </div>
          <div className="flex flex-col md:flex-row items-center justify-center gap-5">
            <Pie_Chart labels={hype_meter_labels} datasets={hype_meter_dataset} options={hype_meter_options} className={hype_meter_design} />
            <div className="flex flex-col gap-2">
              <h3 className="text-center sm:text-md lg:text-lg font-semibold">Overall: {"  "}
                <span className="sm:text-xl lg:text-3xl text-lg">50/100</span>
              </h3>
              <div className="sm:text-xl lg:text-2xl text-lg">
                <h4><span role="img" aria-label="grinning face">😀</span> : 50</h4>
                <h4> <span role="img" aria-label="grinning face">😣</span> : 20</h4>
                <h4> <span role="img" aria-label="grinning face">😐</span> : 30</h4>
              </div>
            </div>
          </div>
        </div>
        <div className="flex flex-col md:flex-row justify-between gap-4 md:mt-4 md:max-w-9/12 lg:max-w-full max-w-full">

          <div className="flex flex-col items-center justify-between border border-black dark:border-white md:w-1/2 bg-secondary dark:bg-tertiary/20 rounded-md">
            <div className="flex flex-row gap-2 pt-2">
              <h3 className="text-center font-semibold text-md md:text-lg lg:text-xl">{Object.keys(meters[1])[0]}</h3>
              <HoverCard>
                <HoverCardTrigger>
                  <IoMdInformationCircleOutline className="mt-[0.25rem] dark:hidden" />
                  <IoMdInformationCircle className="mt-[0.25rem] invisible dark:visible dark:block" />
                </HoverCardTrigger>
                <HoverCardContent>
                {meters[1]["Disruption Score"]}
                </HoverCardContent>
              </HoverCard>
            </div>
            <div className="w-full h-full">
              <GaugeComponent style={{ width: '100%', height: '100%' }} value={disruption_score} type={"radial"} labels={{
                valueLabel: {
                  style: { fill: "var(--tick-label-color)" }

                },
                tickLabels: {
                  type: "inner",
                  ticks: [
                    { value: 20 },
                    { value: 40 },
                    { value: 60 },
                    { value: 80 },
                    { value: 100 }
                  ],

                  defaultTickValueConfig: {
                    style: { fill: "var(--tick-label-color)" }

                  },
                }
              }}
                arc={{
                  colorArray: ['#5BE12C', '#EA4228'],
                  subArcs: [{ limit: 20 }, {}, {}, {}, {}],
                  padding: 0.02,
                  width: 0.2
                }}
                pointer={{
                  elastic: true,
                  animationDelay: 0,
                  color: '#000000',
                }}
              />
            </div>
          </div>
          <div className="flex flex-col items-center justify-between border border-black dark:border-white md:w-1/2 bg-secondary dark:bg-tertiary/20 rounded-md">
            <div className="flex flex-row gap-2 pt-2">
              <h3 className="text-center font-semibold text-md md:text-lg lg:text-xl">{Object.keys(meters[2])[0]}</h3>
              <HoverCard>
                <HoverCardTrigger>
                  <IoMdInformationCircleOutline className="mt-[0.25rem] dark:hidden" />
                  <IoMdInformationCircle className="mt-[0.25rem] invisible dark:visible dark:block" />
                </HoverCardTrigger>
                <HoverCardContent>
                {meters[2]["Impact Factor"]}
                </HoverCardContent>
              </HoverCard>
            </div>
            <div className="w-full h-full">
              <GaugeComponent style={{ width: '100%', height: '100%' }} value={impact_factor} type={"radial"} labels={{
                valueLabel: {
                  style: { fill: "var(--tick-label-color)" }
                },
                tickLabels: {
                  type: "inner",
                  ticks: [
                    { value: 20 },
                    { value: 40 },
                    { value: 60 },
                    { value: 80 },
                    { value: 100 }
                  ],

                  defaultTickValueConfig: {
                    style: { fill: "var(--tick-label-color)" }

                  },
                }
              }}
                arc={{
                  colorArray: ['#5BE12C', '#EA4228'],
                  subArcs: [{ limit: 20 }, {}, {}, {}, {}],
                  padding: 0.02,
                  width: 0.2
                }}
                pointer={{
                  elastic: true,
                  animationDelay: 0,
                  color: '#000000',
                }}
              />
            </div>
          </div>
        </div>
      </div>
      <div>
      <Stock_Chart ticker={ticker ?? ""} />

      </div>
    </div>
  );
}
