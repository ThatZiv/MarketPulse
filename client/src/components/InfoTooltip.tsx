import { Info, LucideProps } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "./ui/tooltip";
import { TooltipArrow } from "@radix-ui/react-tooltip";
import React from "react";

interface InfoTooltipProps {
  className?: string;
  Icon?: React.ForwardRefExoticComponent<
    Omit<LucideProps, "ref"> & React.RefAttributes<SVGSVGElement>
  >;
  side?: "top" | "bottom" | "left" | "right";
  size?: "sm" | "md" | "lg";
  children: React.ReactNode | string;
}

export default function InfoTooltip({
  className,
  children,
  side,
  size = "sm",
  Icon,
}: InfoTooltipProps) {
  const sizeMap = {
    sm: "h-3 w-3",
    md: "h-4 w-4",
    lg: "h-5 w-5",
  };
  return (
    <Tooltip>
      <TooltipTrigger asChild className="flex items-center cursor-pointer">
        {Icon ? (
          <Icon className={`${sizeMap[size]} ${className}`} />
        ) : (
          <Info className={`${sizeMap[size]} ${className}`} />
        )}
      </TooltipTrigger>
      <TooltipContent side={side}>
        <span className="text-sm">{children}</span>
        <TooltipArrow />
      </TooltipContent>
    </Tooltip>
  );
}
