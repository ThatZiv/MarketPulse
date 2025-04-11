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
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

const dialogStyle =
  "w-full dark:text-black max-h-[300px] overflow-y-auto bg-slate-300 p-2 rounded-sm";

interface IGenerateStockLLM {
  ticker?: string;
}

export function GenerateStockLLM({ ticker }: IGenerateStockLLM) {
  const api = useApi();
  const [open, setOpen] = React.useState(false);
  const streamingDiv = React.useRef<HTMLDivElement | null>(null);
  const [status, setStatus] = React.useState<
    "streaming" | "done" | "error" | "idle" | "loading"
  >("idle");
  const [isThinking, setIsThinking] = React.useState<null | boolean>(null);
  const [streamData, setStreamData] = React.useState<string>("");
  const formattedStreamData = React.useMemo<string[]>(
    () =>
      streamData
        .split("</think>")
        .map((data) => markdown.parse(data)) as string[],
    [streamData]
  );
  const dontRegenerateStates = ["done", "streaming", "loading"];
  React.useEffect(() => {
    async function generateAIOutput() {
      if (ticker && open) {
        if (dontRegenerateStates.includes(status)) {
          // dont let it run > 1
          return;
        }
        if (status === "error") {
          setStreamData("");
        }
        try {
          setStatus("loading");
          setIsThinking(true); // for deepseek model
          await api?.getStockLlmOutput(ticker, (chunk: string) => {
            setStatus("streaming");
            if (chunk.includes("</think>")) {
              setIsThinking(false);
            }
            setStreamData((prev) => prev + chunk);
            if (streamingDiv.current) {
              // auto scroll bottom
              streamingDiv.current.scrollTop =
                streamingDiv.current.scrollHeight;
            }
          });
        } catch (error) {
          console.error(error);
          setStreamData("Error generating AI output");
          setStatus("error");
        } finally {
          setStatus("done");
        }
      }
    }

    generateAIOutput();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);
  React.useEffect(() => {
    if (!dontRegenerateStates.includes(status)) {
      // dont reset if its still running
      setStreamData("");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  React.useEffect(() => {
    if (!open) {
      setStreamData("");
      setStatus("idle");
      setIsThinking(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker]);
  return (
    <Drawer open={open} onOpenChange={(isOpen) => setOpen(isOpen)}>
      <DrawerTrigger asChild>
        <Button className="w-full flex justify-center items-center">
          <FaWandMagicSparkles /> MarketPulse AI
        </Button>
      </DrawerTrigger>
      <DrawerContent>
        <div className="mx-auto w-full max-w-xl">
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
              {isThinking ? (
                <Accordion type="single" collapsible>
                  <AccordionItem
                    value="item-1"
                    className="bg-slate-200 rounded-sm"
                  >
                    <AccordionTrigger className="flex text-dark  items-center justify-between px-2 animate-pulse">
                      Thinking...
                    </AccordionTrigger>
                    <AccordionContent className="w-full dark:text-black max-h-[300px] overflow-y-auto bg-slate-300 p-2 rounded-sm">
                      <p
                        dangerouslySetInnerHTML={{
                          __html: formattedStreamData[0],
                        }}
                      />
                      {isThinking && <Spinner />}
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              ) : (
                <div ref={streamingDiv} className={dialogStyle}>
                  <p
                    // className={` ${isStreaming ? "animate-slide-in-right" : ""}`}
                    // TODO: figure out how to animate each individual token coming in
                    dangerouslySetInnerHTML={{ __html: formattedStreamData[1] }}
                  />
                  {status === "loading" && <Spinner />}
                </div>
              )}
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
