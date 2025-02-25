import * as React from "react";
import { FaWandMagicSparkles } from "react-icons/fa6";
import { Button } from "@/components/ui/button";
import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerDescription,
  DrawerFooter,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer";
import { Link } from "react-router";
import { useApi } from "@/lib/ApiProvider";
import { Spinner } from "../ui/spinner";
import markdown from "@/lib/markdown";

interface IGenerateStockLLM {
  ticker?: string;
}

export function GenerateStockLLM({ ticker }: IGenerateStockLLM) {
  const api = useApi();
  const [open, setOpen] = React.useState(false);
  const streamingDiv = React.useRef<HTMLDivElement | null>(null);
  const [isStreaming, setIsStreaming] = React.useState(false);
  const [streamData, setStreamData] = React.useState<string>("");
  const formattedStreamData = React.useMemo(
    () => markdown.parse(streamData),
    [streamData]
  );

  React.useEffect(() => {
    async function generateAIOutput() {
      if (ticker && open) {
        try {
          if (isStreaming) {
            // dont let it run > 1
            // FIXME: still will run twice/continue running when drawer closes
            return;
          }
          setIsStreaming(true);
          await api?.getStockLlmOutput(ticker, (chunk: string) => {
            setIsStreaming(true);
            setStreamData((prev) => prev + chunk);
            if (streamingDiv.current) {
              // auto scroll bottom
              streamingDiv.current.scrollTop =
                streamingDiv.current.scrollHeight;
            }
            setIsStreaming(false);
          });
        } catch (error) {
          console.error(error);
          setStreamData("Error generating AI output");
        } finally {
          setIsStreaming(false);
        }
      }
    }

    generateAIOutput();

    return () => {
      // setStreamData("");
      setIsStreaming(false);
    };
  }, [open]);

  return (
    <Drawer open={open} onOpenChange={(isOpen) => setOpen(isOpen)}>
      <DrawerTrigger asChild>
        <Button className="w-full flex justify-center items-center">
          <FaWandMagicSparkles /> MarketPulse AI
        </Button>
      </DrawerTrigger>
      <DrawerContent>
        <div className="mx-auto w-full max-w-lg">
          <DrawerHeader>
            <DrawerTitle>MarketPulse AI</DrawerTitle>
            <DrawerDescription>
              <span>Please understand that AI can make mistakes.</span>{" "}
              <Link className="link" to="/documentation/disclaimer">
                More info
              </Link>
            </DrawerDescription>
          </DrawerHeader>
          <div className="p-4 pb-0">
            <div className="flex items-center justify-center space-x-2"></div>
            <div className="mt-1 min-h-[120px]">
              <div
                ref={streamingDiv}
                className="w-full dark:text-black max-h-[300px] overflow-y-auto bg-slate-300 p-2 rounded-sm"
              >
                <p
                  // className={` ${isStreaming ? "animate-slide-in-right" : ""}`}
                  // TODO: figure out how to animate each individual token coming in
                  dangerouslySetInnerHTML={{ __html: formattedStreamData }}
                />
                {isStreaming && <Spinner />}
              </div>
            </div>
          </div>
          <DrawerFooter>
            <DrawerClose asChild>
              <Button>Close</Button>
            </DrawerClose>
          </DrawerFooter>
        </div>
      </DrawerContent>
    </Drawer>
  );
}
