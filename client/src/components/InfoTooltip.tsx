import { Info } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";
import { TooltipArrow } from "@radix-ui/react-tooltip";

interface InfoTooltipProps {
  className?: string;
  side?: "top" | "bottom" | "left" | "right";
  children: React.ReactNode | string;
}

export default function InfoTooltip({
  className,
  children,
  side,
}: InfoTooltipProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Info className={`h-4 w-3 ${className}`} />
      </TooltipTrigger>
      <TooltipContent side={side}>
        <p className="text-sm">{children}</p>
        <TooltipArrow />
      </TooltipContent>
    </Tooltip>
  );
}
