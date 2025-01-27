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
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";

import { Link } from "react-router-dom";
import { useEffect } from "react";
import { useSupabase } from "@/database/SupabaseProvider";

type googleResponse = {
  clientId: string;
  client_id: string;
  credential: string;
  select_by: string;
};

export function LoginForm({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"div">) {
  const { signInWithEmail, signInWithGoogle } = useSupabase();

  const formSchema = z.object({
    email: z.string().max(50).email(),
    password: z.string().min(8).max(50).regex(/[A-Z]/, { message: "Password must contain at least one uppercase letter" })
      .regex(/[0-9]/, { message: "Password must contain at least one number" })
      .regex(/[\W_]/, { message: "Password must contain at least one special character" }),
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

  // This appears to work though throws errors in the browser?
  // It is in test mode so emails need to be pre-approved
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
    <div className={cn("flex flex-col gap-6", className)} {...props}>
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">Login</CardTitle>
          <CardDescription>
            Enter your email below to login to your account
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
                    <FormLabel className="text-left">Email Address</FormLabel>
                    <FormControl>
                      <Input placeholder="Email Address" required {...field} />
                    </FormControl>
                    <FormDescription></FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="password"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Password</FormLabel>
                    <FormControl>
                      <Input
                        placeholder="Password"
                        type="password"
                        required
                        {...field}
                      />
                    </FormControl>
                    <FormDescription></FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <Button type="submit">Submit</Button>
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
              <span className="underline underline-offset-4">
                <Link to="/create">Sign up</Link>
              </span>
            </div>
          </Form>
        </CardContent>
      </Card>
    </div>
  );
}
