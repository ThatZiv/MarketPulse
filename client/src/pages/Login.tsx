import { LoginForm } from "@/components/login-form";
// import { NavBar } from "@/components/nav-bar";
import { ModeToggle } from "@/components/ui/mode-toggle";
import { useState } from "react";
import { CreateForm } from "@/components/create_user_form";

export default function Login() {
  const [pageState , setPageState] = useState("login"); // "login" or "signup"

  const togglePageState = () => {
    setPageState((prevMode) => (prevMode === "login" ? "signup" : "login"));
  };
  return (
    <div className="m-0 p-0 min-h-screen bg-center bg-no-repeat bg-stocks-image">
      <div className="flex items-center justify-center text-2xl font-bold bg-gradient-to-r from-primary to-secondary text-transparent p-2">
        <div className="flex-1 flex justify-center space-x-2">
          <span className="text-primary font-Abril_Fatface">Market</span>
          <span className="text-tertiary font-Abril_Fatface">Pulse</span>
        </div>
        <ModeToggle />

      </div>
      <div className="md:max-w-lg max-w-xs mx-auto py-8 " style={{
      transform: `rotateY(${pageState === "login" ? 0 : 180}deg)`,
      transitionDuration: "750ms",
      transformStyle: "preserve-3d",
    }}>
      {pageState === "login" ? (
        <LoginForm togglePageState={togglePageState} pageState={pageState}/>
      
      ):(
        <CreateForm togglePageState={togglePageState} pageState={pageState}/>
      )}
      </div>
      
    </div>
  );
}
