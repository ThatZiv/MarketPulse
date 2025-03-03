import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select"; // Adjust the import as needed

interface DropdownMenuDemoProps {
  values: string[];
  selectedValue: string;
  onChange: (value: string) => void;
}

export default function TimeFrameDropdownMenu({ values, selectedValue, onChange }: DropdownMenuDemoProps) {
  return (
    <Select value={selectedValue} onValueChange={onChange}>
      <SelectTrigger className="w-[160px] rounded-lg sm:ml-auto" aria-label="Select a value">
        <SelectValue placeholder="" />
      </SelectTrigger>
      <SelectContent className="rounded-xl">
        {values.map((value: string) => (
          <SelectItem key={value} value={value} className="rounded-lg">
            {value}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
