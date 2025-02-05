import { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import axios from "axios";

import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Title,
} from "chart.js";

ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Title);

type stock_Data = {
  sentiment_data: number;
  stock_close: number;
  stock_high: number;
  stock_id: number;
  stock_low: number;
  stock_open: number;
  stock_volume: number;
  time_stamp: string[];
}[];

type props = { ticker: string };
export default function Stock_Chart(props: props) {
  const [stockData, setStockData] = useState<stock_Data>([]);

  const length = stockData.length;

  useEffect(() => {
    const authToken = localStorage.getItem(
      "sb-xskipmrkpwewdbttinhd-auth-token"
    );
    if (authToken) {
      const token = JSON.parse(authToken);
      //Prevent refetching of data after a chart has already been made
      if (stockData.length == 0) {
        const chartData = async () => {
          await axios
            .get("http://127.0.0.1:5000/stockchart/?ticker=" + props.ticker, {
              method: "get",
              headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${token.access_token}`,
              },
            })
            .then((response) => {
              console.log(response);
              setStockData(response.data);
            })
            .catch((error) => console.error("Error:", error));
        };
        chartData();
      }
    } else {
      console.log("Failed to find");
    }
  }, []);
  if (stockData.length > 0) {
    const options = {
      plugins: {
        title: {
          display: true,
          text: "Stock Data for " + props.ticker,
          font: {
            size: 20,
          },
          color: 'red',
        },
        legend:{
          labels: {
            color: 'red',
          },
        }
        
      },
    };
    const data = {
      labels: [
        stockData[length - 15].time_stamp[0],
        stockData[length - 14].time_stamp[0],
        stockData[length - 13].time_stamp[0],
        stockData[length - 12].time_stamp[0],
        stockData[length - 11].time_stamp[0],
        stockData[length - 10].time_stamp[0],
        stockData[length - 9].time_stamp[0],
        stockData[length - 8].time_stamp[0],
        stockData[length - 7].time_stamp[0],
        stockData[length - 6].time_stamp[0],
        stockData[length - 5].time_stamp[0],
        stockData[length - 4].time_stamp[0],
        stockData[length - 3].time_stamp[0],
        stockData[length - 2].time_stamp[0],
        stockData[length - 1].time_stamp[0],
      ],
      datasets: [
        {
          label: "Stock Price",
          data: [
            stockData[length - 15].stock_close,
            stockData[length - 14].stock_close,
            stockData[length - 13].stock_close,
            stockData[length - 12].stock_close,
            stockData[length - 11].stock_close,
            stockData[length - 10].stock_close,
            stockData[length - 9].stock_close,
            stockData[length - 8].stock_close,
            stockData[length - 7].stock_close,
            stockData[length - 6].stock_close,
            stockData[length - 5].stock_close,
            stockData[length - 4].stock_close,
            stockData[length - 3].stock_close,
            stockData[length - 2].stock_close,
            stockData[length - 1].stock_close,
          ],
          borderColor: "red",
          tension: 0.1,
        },
      ],
    };
    return (
      <div className="border border-black dark:border-white p-4 bg-tertiary/50 dark:bg-tertiary/20 rounded-md w-full">
        <Line data={data} options={options} />
        <a href="https://www.yahoo.com/?ilc=401" target="_blank">
          {" "}
          <img
            src="https://poweredby.yahoo.com/poweredby_yahoo_h_purple.png"
            width="134"
            height="20"
          />{" "}
        </a>
      </div>
    );
  } else {
    return <></>;
  }
}
