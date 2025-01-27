import { LoginForm } from "@/components/login-form";

export default function Login() {
  return (
    <div className="m-0 p-0 h-screen bg-center bg-no-repeat bg-stocks-image">
      <div className="text-2xl font-bold bg-gradient-to-r from-primary to-secondary text-transparent p-2">
        <span className="text-primary font-Abril_Fatface">Market</span>
        <span className="text-tertiary font-Abril_Fatface">Pulse</span>
      </div>

      <div className="md:max-w-lg max-w-sm mx-auto mt-8">
        <LoginForm />
      </div>
    </div>
  );
}
