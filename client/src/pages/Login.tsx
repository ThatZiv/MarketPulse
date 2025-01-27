import { LoginForm } from "@/components/login-form";

export default function Login() {
  return (
    <div className="mt-0">
      <div className="text-2xl font-bold">
        <span className="text-primary">Market</span>
        <span className="text-tertiary">Pulse</span>
      </div>

      <div className="md:max-w-sm mx-auto">
        <LoginForm />
      </div>
    </div>
  );
}
