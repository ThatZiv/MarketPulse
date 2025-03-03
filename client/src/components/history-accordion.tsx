import { useState, useEffect } from "react"
import {
    Accordion,
    AccordionContent,
    AccordionItem,
    AccordionTrigger,
} from "@/components/ui/accordion"
import { type Stock } from "@/types/stocks";
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
import { Button } from "./ui/button";

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
    const { user, supabase } = useSupabase();

    const {
        value: stocks,
        error: stocksError,
        loading: stocksLoading,
      } = useAsync<Stock[]>(
        () =>
          new Promise((resolve, reject) => {
            supabase
              .from("Stocks")
              .select("*")
              .then(({ data, error }) => {
                if (error) reject(error);
                return resolve(data || []);
              });
          }),
        [supabase]
      );
      const stockid = stocks?.filter(stock => stock.stock_ticker === ticker).map(stock => stock.stock_id);
    
      console.log(Number(stockid));
    
    const { value: history, error: historyError } = useAsync<PurchaseHistoryResponse[]>(
        () =>
          new Promise((resolve, reject) => {
            supabase
              .from("User_Stock_Purchases")
              .select("date, price_purchased, amount_purchased")
              .eq("user_id", user?.id)
              .eq("stock_id", 5)
              .limit(10)
              .then(({ data, error }) => {
                if (error) reject(error);
                resolve(data || []);
              });
          }),
        [user, supabase]
      );
      console.log(history);

    return (
        <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="item-1">
                <AccordionTrigger className="text-2xl justify-center">Investment History</AccordionTrigger>
                <AccordionContent>
                    <Table>
                        <TableCaption>A list of your investments.</TableCaption>
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
                            {history?.map((row) => (
                                <TableRow key={row.date}>
                                    <TableCell className="font-medium">{new Date(row.date).toLocaleDateString('en-US')}</TableCell>
                                    <TableCell>{row.amount_purchased}</TableCell>
                                    <TableCell>{row.price_purchased}</TableCell>
                                    <TableCell className="">{124}</TableCell>
                                    <TableCell className="">
                                        <Button variant={'delete'}>Delete</Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </AccordionContent>
            </AccordionItem>
        </Accordion>
    )
}


