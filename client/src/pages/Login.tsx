import { LoginForm } from "@/components/login-form";

export default function Login() {
  return (
    <div className="m-0 p-0 bg-stocks-image">
      <div className="text-2xl font-bold bg-gradient-to-r from-primary to-secondary text-transparent p-4">
        <span className="text-primary font-Abril_Fatface">Market</span>
        <span className="text-tertiary font-Abril_Fatface">Pulse</span>
      </div>

      <div className="md:max-w-sm mx-auto my-10">
        <LoginForm />
      </div>
    </div>
  );
}
