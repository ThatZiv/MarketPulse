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
import { useSupabase } from "@/database/SupabaseProvider";
import { useState } from "react";
import { FcGoogle } from "react-icons/fc"; // Google Icon

interface CreateFormProps extends React.ComponentPropsWithoutRef<"div"> {
  pageState: string;
  togglePageState: () => void;
}
export function CreateForm({
  className,
  pageState,
  togglePageState,
  ...props
}: CreateFormProps) {
  const { signUpNewUser } = useSupabase();
  const [isFlipped, setIsFlipped] = useState(false);
  const formSchema = z
    .object({
      email: z.string().min(2).max(50),
      password: z.string().min(8).max(50),
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

  return (
    <div className={cn("flex flex-col gap-6", className)} {...props} style={{
      transform: `rotateY(${isFlipped ? 180 : 0}deg)`,
      transitionDuration: "250ms",
      transformStyle: "preserve-3d",
    }}>
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">Create Account</CardTitle>
          <CardDescription>
            Enter email and password for your account
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8" >
              <FormField
                control={form.control}
                name="email"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Email Address</FormLabel>
                    <FormControl>
                      <Input placeholder="Email Address" {...field} />
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
              <FormField
                control={form.control}
                name="password2"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Confirm Password</FormLabel>
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
              <Button className="dark:text-white" type="submit">Create</Button>
              <div className="flex items-center my-4">
                <div className="w-full h-px bg-gray-300"></div>
                <span className="px-4 text-gray-500 text-sm">OR</span>
                <div className="w-full h-px bg-gray-300"></div>
              </div>
            </form>
            <button className="flex items-center justify-between w-full py-2 my-3 px-4 bg-white border border-gray-300 rounded-lg shadow-md hover:bg-gray-100 transition duration-200">
              <div className="mr-2"> <FcGoogle size={20} /></div>
              
              <span className="text-gray-700 flex-grow text-center">Sign up with Google</span> {/* Text centered */}
            </button>
            <div className="mt-4 text-center text-sm">
              Already have an account?{" "}
              <span className="underline underline-offset-4">
                <button className="underline" onClick={() => {
                  togglePageState();  // Toggle login/signin state
                  setIsFlipped((prev) => !prev);  // Flip the UI element
                }}
                >Login</button>
              </span>
            </div>
          </Form>
        </CardContent>
      </Card>
    </div>
  );
}
