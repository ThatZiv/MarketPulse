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
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { useSupabase } from "@/database/SupabaseProvider";
import { useState } from "react";
import { Eye, EyeOff } from "lucide-react";
import { useNavigate } from "react-router";
import { toast } from "sonner";

export function ResetPasswordForm() {
  const [password, setPassword] = useState("");
  const [, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const formSchema = z
    .object({
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
      password: "",
      password2: "",
    },
  });

  const navigate = useNavigate();
  const { supabase } = useSupabase();

  async function onSubmit(values: z.infer<typeof formSchema>) {
    toast("Are you sure you want to reset your password?", {
      action: {
        label: "Confirm",
        onClick: async () => {
          const { error } = await supabase.auth.updateUser({
            password: values.password,
          });

          if (error) {
            toast.error("Failed reseting your password", {
              description: error.message,
            });
          } else {
            toast.success("Password reset successful!");
            navigate("/");
          }
        },
      },
    });
  }
  const passwordValidations = [
    { text: "At least 8 characters", isValid: password.length >= 8 },
    { text: "At least 1 uppercase letter", isValid: /[A-Z]/.test(password) },
    { text: "At least 1 number", isValid: /[0-9]/.test(password) },
    { text: "At least 1 special character", isValid: /[\W_]/.test(password) },
  ];

  return (
    <div className={cn("flex flex-col gap-6")}>
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">Reset Password</CardTitle>
          <CardDescription>
            Enter a new password for your account
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
              <FormField
                control={form.control}
                name="password"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Password</FormLabel>
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
                    <FormLabel>Confirm Password</FormLabel>
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
                Reset
              </Button>
            </form>
          </Form>
        </CardContent>
      </Card>
    </div>
  );
}
