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
import { AlertCircle } from "lucide-react"
import { IoCloseOutline } from "react-icons/io5";
import {
    Alert,
    AlertDescription,
    AlertTitle,
} from "@/components/ui/alert"
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
};
export function DeleteStock({ ticker, stock_id }: DeleteStockProps) {
    const { user, supabase } = useSupabase();
    const queryClient = useQueryClient();
    const [inputValue, setInputValue] = useState("");
    const navigate = useNavigate();

    const mutation = useMutation({
        mutationFn: deleteStock,
        onSuccess: async () => {
            await queryClient.invalidateQueries({
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
            await queryClient.invalidateQueries({
                queryKey: [cache_keys.USER_STOCKS],
            });
            await navigate("/", { replace: true });
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
                <AlertDialogHeader className="flex justify-between items-start">
                    <div className="flex justify-end w-full">
                        <AlertDialogCancel className="flex justify-end w-fit" onClick={() => { setInputValue(""); }}><IoCloseOutline /></AlertDialogCancel>
                    </div>
                    <AlertDialogTitle>Are you absolutely sure you want to delete the entire stock history?</AlertDialogTitle>
                    <AlertDialogDescription>
                        <Alert variant="destructive">
                            <AlertCircle className="h-4 w-4" />
                            <AlertTitle>Warning</AlertTitle>
                            <AlertDescription>
                                This action cannot be undone. This will permanently delete your
                                stock history and remove your data.
                            </AlertDescription>
                        </Alert>
                        <Alert variant="default" className="mt-4">
                            <AlertTitle> Deleting this stock results in:</AlertTitle>
                            <AlertDescription>
                                <ul className="list-disc pl-5">
                                    <li>Your Stock purchase history will be permanently deleted.</li>
                                    <li>Stock predictions will be lost.</li>
                                    <li>All stored data for {ticker} stock cannot be retrieved.</li>
                                    <li>To re-add {ticker} stock, you'll need to enter the stock details again.</li>
                                </ul>
                            </AlertDescription>
                        </Alert>
                    </AlertDialogDescription>
                    <h2 className="text-md pt-3 border-t-2">Please enter <span className="font-medium">{ticker} </span>to confirm the deletion is intentional.</h2>
                    <Input type="text" placeholder={ticker} value={inputValue} onChange={(e) => setInputValue(e.target.value)} />
                </AlertDialogHeader>
                <AlertDialogFooter>
                    <AlertDialogAction onClick={handleDelete} className="bg-[#e50000] text-[#e50000]-foreground shadow
                     hover:bg-[#e50000]/90 text-white border-1dark:hover:border-white hover:border-black hover:border-2
                      dark:active:bg-[#e50000]/40 active:bg-[#e50000]/40 w-full">I understand the consequences of removing this stock.</AlertDialogAction>
                </AlertDialogFooter>
            </AlertDialogContent>
        </AlertDialog>
    )
}
