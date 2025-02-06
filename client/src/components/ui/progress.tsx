import * as React from "react";
import * as ProgressPrimitive from "@radix-ui/react-progress";
import { cn } from "@/lib/utils";

const Progress = React.forwardRef<
  React.ElementRef<typeof ProgressPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ProgressPrimitive.Root>
>(({ className, value = 0, ...props }, ref) => {
  const remaining = 100 - (value || 0); // Calculate the remaining value

  return (
    <ProgressPrimitive.Root
      ref={ref}
      className={cn(
        "relative h-5 w-full overflow-hidden rounded-full",
        className
      )}
      {...props}
    >
      {/* Red part with hover effect */}
      <div
        className="absolute right-0 top-0 h-full bg-red-600 w-full transition-all hover:bg-red-800"
        style={{ width: `${remaining}%` }} 
      />
      <ProgressPrimitive.Indicator
        className="h-full w-full bg-green-600 transition-all hover:bg-green-800"
        style={{ transform: `translateX(-${remaining}%)` }} 
      />
    </ProgressPrimitive.Root>
  );
});

Progress.displayName = ProgressPrimitive.Root.displayName;

export { Progress };
