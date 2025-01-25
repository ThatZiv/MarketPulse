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

import { signInWithEmail, supabase } from "@/database/supabase";
import { Link } from "react-router-dom";
import { useEffect } from "react";

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
  const formSchema = z.object({
    email: z.string().min(2).max(50),
    password: z.string().min(2).max(50),
  });

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });

  function onSubmit(values: z.infer<typeof formSchema>) {
    // Not sure how to handle bad responses will add once decided
    try {
      const response = signInWithEmail(values.email, values.password);
      // Will remove after deciding what to do with responses
      console.log(response);
    } catch (error) {
      console.log("Error on login", error);
    }
  }

  // This appears to work though throws errors in the browser?
  // It is in test mode so emails need to be pre-approved
  window.handleSignInWithGoogle = async (response: googleResponse) => {
    console.log("Callback fired! Response:", response);
    const { data, error } = await supabase.auth.signInWithIdToken({
      provider: "google",
      token: response.credential,
    });
    console.log("Supabase Login", data);
    console.log("Supabase Error", error);
  };

  useEffect(() => {
    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.defer = true;
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
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
              <FormField
                control={form.control}
                name="email"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Email</FormLabel>
                    <FormControl>
                      <Input placeholder="Email" {...field} />
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
                        {...field}
                      />
                    </FormControl>
                    <FormDescription></FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <Button type="submit">Submit</Button>
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
