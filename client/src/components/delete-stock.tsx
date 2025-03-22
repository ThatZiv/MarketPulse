import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
    AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { IoTrash } from "react-icons/io5";
import { Input } from "@/components/ui/input"
import { useState } from "react";
import { toast } from "sonner";
import { useSupabase } from "@/database/SupabaseProvider";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { cache_keys } from "@/lib/constants";
import { useNavigate } from "react-router";
import { SupabaseClient } from "@supabase/supabase-js";

interface DeleteStockProps {
    ticker?: string;
    stock_id?: number;
}
const deleteStock = async ({ stock_id, user_id, supabase }: { stock_id: number, user_id: string, supabase: SupabaseClient }): Promise<void> => {
    const { error: purchasesError } = await supabase
        .from('User_Stock_Purchases')
        .delete()
        .match({ user_id: user_id, stock_id: stock_id });
    if (purchasesError) {
        console.error('Error deleting records:', purchasesError.message);
    }
    const { error: stocksError } = await supabase
        .from('User_Stocks')
        .delete()
        .match({ user_id: user_id, stock_id: stock_id });

    if (stocksError) {
        console.error('Error deleting from User_Stocks:', stocksError.message);
    }
    console.log('Stock successfully deleted');
};
export function DeleteStock({ ticker, stock_id }: DeleteStockProps) {
    const { user, supabase } = useSupabase();
    const queryClient = useQueryClient();
    const [inputValue, setInputValue] = useState("");
    const navigate = useNavigate();

    const mutation = useMutation({
        mutationFn: deleteStock,
        onSuccess: () => {
            queryClient.invalidateQueries({
                queryKey: [cache_keys.USER_STOCKS],
            });
            toast.success("Stock deleted successfully.");
        },
        onError: (error) => {
            console.error("Error deleting stock:", error.message);
            toast.error("Error deleting stock. Please try again.");
        },
    });
    const handleDelete = async () => {
        if (inputValue === ticker) {
            if (stock_id === undefined || stock_id === null) {
                console.error("Invalid stock_id");
                toast.error("Invalid stock selection.");
                return;
            }

            if (!user?.id) {
                console.error("User is not authenticated");
                toast.error("You must be logged in to delete stocks.");
                return;
            }
            mutation.mutate({ stock_id: stock_id ?? 0, user_id: user?.id ?? '', supabase });
            navigate("/", { replace: true });
        } else {
            toast.error("Stock name does not match. Please enter the correct stock name to delete.");
            setInputValue("");
        }
    };

    return (
        <AlertDialog>
            <AlertDialogTrigger asChild>
                <Button variant="delete" size="sm">  <IoTrash className="" /> Delete</Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
                <AlertDialogHeader>
                    <AlertDialogTitle>Are you absolutely sure you want to delete the entire stock history?</AlertDialogTitle>
                    <AlertDialogDescription>
                        This action cannot be undone. This will permanently delete your
                        stock history and remove your data.
                    </AlertDialogDescription>
                    <h2 className="text-md">Please enter the name of the stock you want to delete permanently.
                        This is to ensure that the deletion is intentional.</h2>
                    <Input type="text" placeholder={ticker} value={inputValue} onChange={(e) => setInputValue(e.target.value)} />
                </AlertDialogHeader>
                <AlertDialogFooter>
                    <AlertDialogCancel onClick={() => { setInputValue(""); }}>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={handleDelete} className="bg-[#e50000] text-[#e50000]-foreground shadow hover:bg-[#e50000]/90 text-white border-1 dark:hover:border-white hover:border-black hover:border-2 dark:active:bg-[#e50000]/40 active:bg-[#e50000]/40">Delete</AlertDialogAction>
                </AlertDialogFooter>
            </AlertDialogContent>
        </AlertDialog>
    )
}
