import { cn } from "@/lib/utils";

function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "animate-pulse dark:bg-white/25 rounded-md bg-primary/50",
        className
      )}
      {...props}
    />
  );
}

export { Skeleton };
