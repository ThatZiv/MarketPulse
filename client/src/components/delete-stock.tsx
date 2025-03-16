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

interface DeleteStockProps {
    ticker?: string;
}
export function DeleteStock({ ticker }: DeleteStockProps) {
    const [inputValue, setInputValue] = useState("");

    const handleDelete = () => {
        if (inputValue === ticker) {
            console.log(inputValue + " deleted " + ticker);
            console.log("Stock deleted");
        } else {
            toast.error("Stock name does not match. Please enter the correct stock name.");
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
