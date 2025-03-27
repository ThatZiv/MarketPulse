
import { type PurchaseHistoryDatapoint } from "@/types/global_state"

function average_share_value(purchases:PurchaseHistoryDatapoint[])
{
let count = 0
let cost = 0
for(let i = 0; i<purchases.length; i++)
{
    if(purchases[i].amount_purchased < 0)
    {   
        //Sold shares are subtracted based on the current average value 
        cost = cost + (purchases[i].amount_purchased*cost/count)
        count = count + purchases[i].amount_purchased
    }
    else
    {
        cost = cost+purchases[i].amount_purchased*purchases[i].price_purchased
        count = count + purchases[i].amount_purchased
    }
}
return {average:cost/count, amount:count}
}
type props = {
    current_price:number, 
    predicted_price:number, 
    purchases:PurchaseHistoryDatapoint[]
}
export function Suggestion(props:props)
{
    const investment_value = average_share_value(props.purchases)

    const high = 1.05
    const low = .95
    
    // Calculate offsets
    const upper_offset  = .025*Math.atan(100*(props.predicted_price-investment_value.average)/props.current_price)
    const lower_offset = .025*Math.atan(100*(props.predicted_price-investment_value.average)/props.current_price)
    
    if(props.predicted_price/props.current_price > high-upper_offset)
    {
        return <div>Buy</div>
    }
    if(props.predicted_price/props.current_price < low-lower_offset)
    {
        return <div>Sell</div>
    }
    else
    {
        return <div>Hold</div>
    }
}