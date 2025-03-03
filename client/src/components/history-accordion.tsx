import { useState, useEffect } from "react"
import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
} from "@/components/ui/accordion"
import {
    Table,
    TableBody,
    TableCaption,
    TableCell,
    TableFooter,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"
import { useSupabase } from "@/database/SupabaseProvider";
import useAsync from "@/hooks/useAsync";

const invoices = [
    {
        invoice: "INV001",
        paymentStatus: "Paid",
        totalAmount: "$250.00",
        paymentMethod: "Credit Card",
    },
    {
        invoice: "INV002",
        paymentStatus: "Pending",
        totalAmount: "$150.00",
        paymentMethod: "PayPal",
    },
    {
        invoice: "INV003",
        paymentStatus: "Unpaid",
        totalAmount: "$350.00",
        paymentMethod: "Bank Transfer",
    },
    {
        invoice: "INV004",
        paymentStatus: "Paid",
        totalAmount: "$450.00",
        paymentMethod: "Credit Card",
    },
    {
        invoice: "INV005",
        paymentStatus: "Paid",
        totalAmount: "$550.00",
        paymentMethod: "PayPal",
    },
    {
        invoice: "INV006",
        paymentStatus: "Pending",
        totalAmount: "$200.00",
        paymentMethod: "Bank Transfer",
    },
    {
        invoice: "INV007",
        paymentStatus: "Unpaid",
        totalAmount: "$300.00",
        paymentMethod: "Credit Card",
    },
]
interface PurchaseHistoryResponse {
    date: string;
    price_purchased: number;
    amount_purchased: number;
}
interface props{
    ticker: string;
}
export default function TransactionHistory({ticker}:props) {
    const { user, supabase } = useSupabase(); // Assuming `useSupabase` is a custom hook that provides `user` and `supabase`

    const {
        value: history,
        error: stocksError,
        loading: stocksLoading,
      } = useAsync<PurchaseHistoryResponse[]>(
        async () => {
          if (!user || !ticker) {
            throw new Error("User or ticker is missing");
          }
    
          const { data: stockData, error: stockError } = await supabase
            .from("stocks")
            .select("stock_id")
            .eq("stock_name", ticker)
            .single();
    
          if (stockError) {
            throw stockError;
          }
    
          const { data: purchaseHistoryData, error: historyError } = await supabase
            .from("User_Stock_Purchases")
            .select("date, price_purchased, amount_purchased")
            .eq("user_id", user?.id)
            .eq("stock_id", stockData?.stock_id);
    
          if (historyError) {
            throw historyError;
          }
    
          return purchaseHistoryData|| [];
        },
        [ticker, user]
      );
    
      useEffect(() => {
        if (history) {
          console.log("Purchase History Data:", history);
        } else {
          console.log("Purchase History Data: No history available");
        }
      }, [history]);
      
    
      if (stocksLoading) {
        return <div>Loading...</div>;
      }
    
      if (stocksError) {
        return (
          <div className="flex flex-col justify-center items-center h-screen">
            <h1 className="text-3xl">Error</h1>
            <p className="text-primary">
              Unfortunately, we encountered an error fetching your history. Please
              refresh the page or try again later.
            </p>
          </div>
        );
      }
    
      if (!history || history.length === 0) {
        return <div>No transaction history available.</div>;
      }
    return (
        <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="item-1">
                <AccordionTrigger className="text-2xl justify-center">Transaction History</AccordionTrigger>
                <AccordionContent>
                    <Table>
                        <TableCaption>A list of your recent invoices.</TableCaption>
                        <TableHeader>
                            <TableRow>
                                <TableHead className="w-[100px] text-center">Timestamp</TableHead>
                                <TableHead className="text-center">Stocks Bought</TableHead>
                                <TableHead className="text-center">Past Stock Price</TableHead>
                                <TableHead className="text-center">Current Stock Price</TableHead>

                                <TableHead className="text-center">Delete?</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {invoices.map((invoice) => (
                                <TableRow key={invoice.invoice}>
                                    <TableCell className="font-medium">{invoice.invoice}</TableCell>
                                    <TableCell>{invoice.paymentStatus}</TableCell>
                                    <TableCell>{invoice.paymentMethod}</TableCell>
                                    <TableCell className="text-right">{invoice.totalAmount}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </AccordionContent>
            </AccordionItem>
        </Accordion>
    )
}


