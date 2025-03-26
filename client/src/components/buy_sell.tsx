type purchases = {
    amount_purchased: number,
    price_purchased:number
}[]
function average_share_value(purchases:purchases)
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
type output = {
    suggestion: string,
    profits:number
}
export function suggestion(current_price:number, predicted_price:number, purchases:purchases)
{
    const investment_value = average_share_value(purchases)

    const output:output = {suggestion:"", profits:(predicted_price - investment_value.average)*investment_value.amount}    
    const high = 1.05
    const low = .95
    
    // Calculate offsets
    const upper_offset  = .025*Math.atan(100*(predicted_price-investment_value.average)/current_price)
    const lower_offset = .025*Math.atan(100*(predicted_price-investment_value.average)/current_price)
    // percentage sell using a funtion?

    // buy confidence using a function?
    if(predicted_price/current_price > high-upper_offset)
    {
        output.suggestion = "Buy"
    }
    if(predicted_price/current_price < low-lower_offset)
    {
        output.suggestion = "Sell"
    }
    else
    {
        output.suggestion="Hold"
    }

return (
<div>
    <h1>{output.suggestion}</h1>
    {output.profits > 0?(
        <div>
            <p>Expected profit </p>\
            <p className="text-green-700">$ {output.profits}</p>
        </div>
    ):(
        <div>
            <p>Expected Loss </p>
            <p className="text-red-700">$ {output.profits}</p>
        </div>
    )}
</div>
);

}