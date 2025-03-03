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
import { useSupabase } from "@/database/SupabaseProvider";
import { useEffect, useState } from "react";
import { Eye, EyeOff } from "lucide-react";

type googleResponse = {
  clientId: string;
  client_id: string;
  credential: string;
  select_by: string;
};
interface CreateFormProps extends React.ComponentPropsWithoutRef<"div"> {
  togglePageState: () => void;
}
export function CreateForm({
  className,
  togglePageState,
  ...props
}: CreateFormProps) {
  const { signUpNewUser } = useSupabase();
  const [isFlipped, setIsFlipped] = useState(true);
  const { signInWithGoogle } = useSupabase();
  const [password, setPassword] = useState("");
  const [, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const formSchema = z
    .object({
      email: z.string().min(2).max(50),
      password: z
        .string()
        .min(8)
        .max(50)
        .regex(/[A-Z]/, {
          message: "Password must contain at least one uppercase letter",
        })
        .regex(/[0-9]/, {
          message: "Password must contain at least one number",
        })
        .regex(/[\W_]/, {
          message: "Password must contain at least one special character",
        }),
      password2: z.string(),
    })
    .refine(
      (data) => {
        return data.password === data.password2;
      },
      {
        message: "Passwords do not match",
        path: ["password2"],
      }
    );

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      email: "",
      password: "",
      password2: "",
    },
  });

  async function onSubmit(values: z.infer<typeof formSchema>) {
    await signUpNewUser(values.email, values.password);
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
  }, []);

  const passwordValidations = [
    { text: "At least 8 characters", isValid: password.length >= 8 },
    { text: "At least 1 uppercase letter", isValid: /[A-Z]/.test(password) },
    { text: "At least 1 number", isValid: /[0-9]/.test(password) },
    { text: "At least 1 special character", isValid: /[\W_]/.test(password) },
  ];

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
          <CardTitle className="text-2xl">Create an Account</CardTitle>
          <CardDescription>
            Enter email and password for your account
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
                    <FormControl>
                      <Input placeholder="Email Address" {...field} />
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
                          {...field}
                          onChange={(e) => {
                            field.onChange(e);
                            setPassword(e.target.value);
                          }}
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
                    <div className="mt-2 text-sm">
                      {passwordValidations.map((req, index) => (
                        <div
                          key={index}
                          className={`flex items-center ${
                            req.isValid ? "text-green-500" : "text-red-500"
                          }`}
                        >
                          {req.isValid ? "✅" : "❌"} {req.text}
                        </div>
                      ))}
                    </div>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="password2"
                render={({ field }) => (
                  <FormItem>
                    <FormControl>
                      <div className="relative">
                        <Input
                          placeholder="Confirm Password"
                          type={showConfirmPassword ? "text" : "password"}
                          {...field}
                          onChange={(e) => {
                            field.onChange(e);
                            setConfirmPassword(e.target.value);
                          }}
                        />
                        <button
                          type="button"
                          onClick={() =>
                            setShowConfirmPassword((prev) => !prev)
                          }
                          className="absolute inset-y-0 right-2 flex items-center text-gray-500"
                        >
                          {showConfirmPassword ? (
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
              <Button className="dark:text-white" type="submit">
                Create
              </Button>
              <div className="flex items-center my-4">
                <div className="w-full h-px bg-gray-300"></div>
                <span className="px-4 text-gray-500 text-sm">OR</span>
                <div className="w-full h-px bg-gray-300"></div>
              </div>
            </form>
            <div
              className="mt-3"
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
              Already have an account?{" "}
              <span className="link ">
                <button
                  onClick={() => {
                    togglePageState();
                    setIsFlipped((prev) => !prev);
                  }}
                >
                  Login
                </button>
              </span>
            </div>
          </Form>
        </CardContent>
      </Card>
    </div>
  );
}
