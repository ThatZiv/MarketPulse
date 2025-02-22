#!/bin/bash

# curl -H "User-Agent: curl" https://query1.finance.yahoo.com/v8/finance/chart/TM\?period1\=1577836800\&period2\=1735257600\&interval\=1d
#     | jq "{close: .chart.result[0].indicators.quote[0].close, open: .chart.result[0].indicators.quote[0].open, volume: .chart.result[0].indicators.quote[0].volume, high: .chart.result[0].indicators.quote[0].high, low: .chart.result[0].indicators.quote[0].low }"

stocks=("TM" "TSLA" "F")
# stocks=("TM")
mkdir -p mockStocks
for stock in "${stocks[@]}"; do
    # from 2020 to 2024
    curl -H "User-Agent: Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36" \
        "https://query1.finance.yahoo.com/v8/finance/chart/$stock?period1=1577836800&period2=1735257600&interval=1d" |
        jq "{ close: .chart.result[0].indicators.quote[0].close, open: .chart.result[0].indicators.quote[0].open, volume: .chart.result[0].indicators.quote[0].volume, high: .chart.result[0].indicators.quote[0].high, low: .chart.result[0].indicators.quote[0].low }" \
            >"mockStocks/$stock.json"
done
