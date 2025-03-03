import { LoginForm } from "@/components/login-form";
// import { NavBar } from "@/components/nav-bar";
import { ModeToggle } from "@/components/ui/mode-toggle";
import { useState } from "react";
import { CreateForm } from "@/components/create_user_form";
import { ResetPassword } from "@/components/forgot-password";


export default function UserAuth() {
  const [pageState, setPageState] = useState("login"); // "login" or "signup"
  const [passwordState, resetPasswordState] = useState(false);
  const togglePageState = () => {
    setPageState((prevMode) => (prevMode === "login" ? "signup" : "login"));
  };

  return (
    <div className="m-0 p-0 min-h-screen bg-center bg-no-repeat bg-cover bg-stocks-graph dark:bg-dark-stocks-graph">
      <div className="flex items-center text-2xl sm:text-5xl font-bold text-transparent p-2">
        <div className="ml-5 pt-2 mb-5">
          <div className="flex sm:justify-center justify-left items-center gap-2">
            <span className="text-white">MarketPulse</span>
            <img
              src="/public/images/MarketPulse_Logo.png"
              alt="MarketPulse Logo"
              className="sm:h-24 sm:w-24 w-12 h-12"
            />
          </div>
        </div>
        <div className="absolute right-0 top-0 p-2">
          <ModeToggle />
        </div>
      </div>
      <div
        className="md:max-w-lg max-w-xs mx-auto pb-8"
        style={{
          transform: `rotateY(${pageState === "login" ? 0 : 180}deg)`,
          transitionDuration: "750ms",
          transformStyle: "preserve-3d",
        }}
      >
        {passwordState ? (
          <ResetPassword resetPasswordState={resetPasswordState} />
        ) : (
          <div>
            {pageState === "login" ? (
              <LoginForm
                togglePageState={togglePageState}
                resetPasswordState={resetPasswordState}
              />
            ) : (
              <CreateForm togglePageState={togglePageState} />
            )}{" "}
          </div>
        )}
      </div>
      {/* <Footer /> */}
    </div>
  );
}
