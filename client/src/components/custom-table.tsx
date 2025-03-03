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

type Predictions = {
    date: string;
    predicted_price: number;
};

interface CustomTableProps {
    caption: string;
    tableheader: string[];
    predictions: Predictions[];
}

export default function CustomTable({ caption, tableheader, predictions }: CustomTableProps) {
    return (
        <Table>
            <TableCaption>{caption}</TableCaption>
            <TableHeader>
                <TableRow>
                    {tableheader.map((header: string) => (
                        <TableHead key={header}>{header}</TableHead>
                    ))}
                </TableRow>
            </TableHeader>
            <TableBody>
                {predictions.map((prediction, index) => (
                    <TableRow key={index} className="hover:bg-gray-100">
                        <TableCell className="text-left">{prediction.date}</TableCell>
                        <TableCell className="text-left">{prediction.predicted_price}</TableCell>
                    </TableRow>
                ))}
            </TableBody>

        </Table>
    );
}


