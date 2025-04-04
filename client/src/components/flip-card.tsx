import React, { ReactNode, useState } from "react";
import { Button } from "./ui/button";

interface FlipCardProps {
  frontContent: ReactNode;
  backContent: ReactNode;
}
//added this flip component for the sentiment scores. Will integrate in prototype-2
const FlipCard: React.FC<FlipCardProps> = ({ frontContent, backContent }) => {
  const [isFlipped, setIsFlipped] = useState(false);
  const [pageState, setPageState] = useState("frontContent");
  const handleFlip = () => {
    setPageState(pageState === "frontContent" ? "backContent" : "frontContent");
    setIsFlipped(!isFlipped);
  };
  return (
    <div
      className="relative w-96 h-96 p-4 perspective"
      style={{
        transform: `rotateY(${isFlipped ? 0 : 180}deg)`,
        transitionDuration: "250ms",
        transformStyle: "preserve-3d",
      }}
    >
      <div
        className={`relative w-full h-full transition-transform duration-500 transform ${isFlipped ? "rotate-y-180" : ""}`}
      >
        {pageState === "frontContent" ? (
          <div className="absolute w-full h-80 backface-hidden flex items-center justify-center bg-secondary rounded-md dark:bg-primary">
            {frontContent}
          </div>
        ) : (
          <div className="absolute w-full h-80 backface-hidden rotate-y-180 flex items-center justify-center bg-white dark:bg-gray-800 rounded-md">
            {backContent}
          </div>
        )}
      </div>
      <Button onClick={handleFlip}>More...</Button>
    </div>
  );
};

export default FlipCard;
