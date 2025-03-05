import { useSupabase } from "@/database/SupabaseProvider";
import { cache_keys } from "@/lib/constants";
import { useQuery } from "@tanstack/react-query";
import { Spinner } from "./ui/spinner";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
} from "@/components/ui/accordion";
import moment from "moment";
interface PurchaseHistoryProps {
  stock_id: number;
}

export function PurchaseHistory({ stock_id }: PurchaseHistoryProps) {
  const { supabase } = useSupabase();
  const { data, isError, isLoading } = useQuery({
    queryKey: [cache_keys.USER_STOCK_PURCHASES, stock_id],
    queryFn: async () => {
      const resp = await supabase
        .from("User_Stock_Purchases")
        .select("*")
        .eq("stock_id", stock_id);
      if (resp.error) {
        throw new Error(resp.error.message);
      }
      return resp.data;
    },
    enabled: !!stock_id && !!supabase,
  });
  if (isLoading) {
    return <Spinner />;
  }
  if (isError) {
    return (
      <div>Failed to fetch your purchase history. Please try again later.</div>
    );
  }
  return (
    <Accordion type="single" collapsible className="w-full">
      <AccordionItem value="purchasehistory" title="Purchase History">
        <AccordionContent>
          <Table>
            <TableCaption>Your purchase history</TableCaption>
            <TableHead>
              <TableRow>
                <TableHeader>Date</TableHeader>
                <TableHeader>Quantity</TableHeader>
                <TableHeader>Price</TableHeader>
                <TableHeader>Total</TableHeader>
              </TableRow>
            </TableHead>
            <TableBody>
              {data?.map((row) => (
                <TableRow key={row.id}>
                  <TableCell>{row.price_purchased}</TableCell>
                  <TableCell>{row.amount_purchased}</TableCell>
                </TableRow>
              ))}
            </TableBody>
            <TableFooter>
              As of {moment(new Date().toLocaleDateString()).calendar()}
            </TableFooter>
          </Table>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
