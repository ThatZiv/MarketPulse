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
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"
import { useSupabase } from "@/database/SupabaseProvider";
import useAsync from "@/hooks/useAsync";
import { Button } from "./ui/button";

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
      const stockid = stocks?.find(stock => stock.stock_ticker === ticker)?.stock_id;
    
    
    const { value: history, error: historyError } = useAsync<PurchaseHistoryResponse[]>(
        () =>
          new Promise((resolve, reject) => {
            supabase
              .from("User_Stock_Purchases")
              .select("date, price_purchased, amount_purchased")
              .eq("user_id", user?.id)
              .eq("stock_id", stockid)
              .limit(10)
              .then(({ data, error }) => {
                if (error) reject(error);
                resolve(data || []);
              });
          }),
        [user, supabase]
      );
      if (stocksError) {
        return (
          <div className="flex flex-col justify-center items-center h-screen">
            <h1 className="text-3xl">Error</h1>
            <p className="text-primary">
              Unfortunately, we encountered an error fetching the stocks. Please
              refresh the page or try again later.
            </p>
          </div>
        );
      }
      if (historyError) {
        return (
          <div className="flex flex-col justify-center items-center h-screen">
            <h1 className="text-3xl">Error Retrieving History</h1>
            <p className="text-primary">
              You have no history for this stock.
            </p>
          </div>
        );
      }
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


