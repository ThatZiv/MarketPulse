import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/ui/form";
import { Eye, EyeOff } from "lucide-react";

import { useEffect, useState } from "react";
import { useSupabase } from "@/database/SupabaseProvider";

type googleResponse = {
  clientId: string;
  client_id: string;
  credential: string;
  select_by: string;
};

interface LoginFormProps extends React.ComponentPropsWithoutRef<"div"> {
  togglePageState: () => void;
  resetPasswordState: (arg: boolean) => void;
}

export function LoginForm({
  className,
  togglePageState,
  resetPasswordState,
  ...props
}: LoginFormProps) {
  const { signInWithEmail, signInWithGoogle } = useSupabase();
  const [isFlipped, setIsFlipped] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const formSchema = z.object({
    email: z.string().max(50).email(),
    password: z.string().min(8).max(50),
  });

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });

  async function onSubmit(values: z.infer<typeof formSchema>, event?: Event) {
    event?.preventDefault();
    await signInWithEmail(values.email, values.password);
  }

  window.handleSignInWithGoogle = async (response: googleResponse) => {
    console.log("Callback fired! Response:", response);
    await signInWithGoogle(response);
  };

  useEffect(() => {
    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    document.body.appendChild(script);
    console.log("load");
  }, []);

  return (
    <div
      className={cn("flex flex-col gap-6", className)}
      {...props}
      style={{
        transform: `rotateY(${isFlipped ? 180 : 0}deg)`,
        transitionDuration: "250ms",
        transformStyle: "preserve-3d",
      }}
    >
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">Log in</CardTitle>
          <CardDescription>
            Enter your info below to login to your account
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit((values) => onSubmit(values, event))}
              className="space-y-8 mb-5"
            >
              <FormField
                control={form.control}
                name="email"
                render={({ field }) => (
                  <FormItem>
                    <FormControl>
                      <Input placeholder="Email Address" required {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="password"
                render={({ field }) => (
                  <FormItem>
                    <FormControl>
                      <div className="relative">
                        <Input
                          placeholder="Password"
                          type={showPassword ? "text" : "password"}
                          required
                          {...field}
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword((prev) => !prev)}
                          className="absolute inset-y-0 right-2 flex items-center text-gray-500"
                        >
                          {showPassword ? (
                            <EyeOff size={20} />
                          ) : (
                            <Eye size={20} />
                          )}
                        </button>
                      </div>
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <Button type="submit" className="mt-3">
                Login
              </Button>
              <div className="text-center text-sm">
                Forgot your{" "}
                <span className="link">
                  <button
                    type="button"
                    onClick={() => resetPasswordState(true)}
                  >
                    password
                  </button>
                </span>{" "}
                ?
              </div>
              <div className="flex items-center my-4">
                <div className="w-full h-px bg-gray-300"></div>
                <span className="px-4 text-gray-500 text-sm">OR</span>
                <div className="w-full h-px bg-gray-300"></div>
              </div>
            </form>

            <div
              id="g_id_onload"
              data-client_id="554299705421-su031i3j82o10cjpnss6b7qnualeparh.apps.googleusercontent.com"
              data-context="signin"
              data-ux_mode="popup"
              data-callback="handleSignInWithGoogle"
              data-auto_prompt="false"
            ></div>

            <div
              className="g_id_signin"
              data-type="standard"
              data-shape="rectangular"
              data-theme="outline"
              data-text="signin_with"
              data-size="large"
              data-logo_alignment="left"
            ></div>

            <div className="mt-4 text-center text-sm">
              Don&apos;t have an account?{" "}
              <span className="text-blue-600 hover:text-blue-900 underline-offset-4">
                <button
                  className=""
                  onClick={() => {
                    togglePageState();
                    setIsFlipped((prev) => !prev);
                  }}
                >
                  Create Account
                </button>
              </span>
            </div>
          </Form>
        </CardContent>
      </Card>
    </div>
  );
}
