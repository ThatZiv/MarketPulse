import { LoginForm } from "@/components/login-form";
// import { NavBar } from "@/components/nav-bar";
import { ModeToggle } from "@/components/ui/mode-toggle";

export default function Login() {
  return (
    <div className="m-0 p-0 h-full bg-center bg-no-repeat bg-stocks-image">
      <div className="flex items-center justify-center text-2xl font-bold bg-gradient-to-r from-primary to-secondary text-transparent p-2">
        <div className="flex-1 flex justify-center space-x-2">
          <span className="text-primary font-Abril_Fatface">Market</span>
          <span className="text-tertiary font-Abril_Fatface">Pulse</span>
        </div>
        <ModeToggle />

      </div>


      <div className="md:max-w-lg max-w-sm mx-auto mt-8">
        <LoginForm />
      </div>
    </div>
  );
}
