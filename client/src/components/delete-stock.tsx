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
import { useGlobal } from "@/lib/GlobalProvider";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { cache_keys } from "@/lib/constants";

interface DeleteStockProps {
    ticker?: string;
    stock_id?: number;
}
export function DeleteStock({ ticker, stock_id }: DeleteStockProps) {
    const { supabase,user } = useSupabase();
    const queryClient = useQueryClient();
    const [inputValue, setInputValue] = useState("");
    const deleteStock = async (): Promise<void> => {
        const { data, error } = await supabase
            .from('User_Stock_Purchases')
            .delete()
            .match({ user_id: user?.id, stock_id: stock_id });
        if (error) {
            console.error('Error deleting records:', error.message);
            throw new Error("Error deleting records");
        }
    };
    const mutation = useMutation({
        mutationFn: deleteStock,
        onSuccess: () => {
            queryClient.invalidateQueries({
                queryKey: [cache_keys.USER_STOCKS],
            });
        },
        onError: (error) => {
            console.error("Error deleting stock:", error.message);
            toast.error("Error deleting stock. Please try again.");
        },
    });
    const handleDelete = async () => {
        if (inputValue === ticker) {
            console.log(inputValue + " deleted " + ticker);
            mutation.mutate();
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
                    <h2 className="text-md">Please enter the name of the stock you want to delete permanently</h2>
                    <Input type="text" placeholder={ticker} value={inputValue} onChange={(e) => setInputValue(e.target.value)} />
                </AlertDialogHeader>
                <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={handleDelete} className="bg-[#e50000] text-[#e50000]-foreground shadow hover:bg-[#e50000]/90 text-white border-1 dark:hover:border-white hover:border-black hover:border-2 dark:active:bg-[#e50000]/40 active:bg-[#e50000]/40">Delete</AlertDialogAction>
                </AlertDialogFooter>
            </AlertDialogContent>
        </AlertDialog>
    )
}
