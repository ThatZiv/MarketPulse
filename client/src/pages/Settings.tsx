import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useSupabase } from "@/database/SupabaseProvider";
import { LockIcon, SaveIcon } from "lucide-react";
import React from "react";
import { toast } from "sonner";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { useTheme } from "@/components/ui/theme-provider";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useNavigate, useParams } from "react-router";
import { Label } from "@/components/ui/label";

export default function SettingsPage() {
  const { supabase, user, signOut } = useSupabase();
  const [state, setState] = React.useState<"loading" | "error" | "done">(
    "loading"
  );
  const { tab: tabParam } = useParams();
  const { setTheme } = useTheme();
  const navigate = useNavigate();
  const [tab, setTab] = React.useState(tabParam ?? "account");
  const accountFormSchema = z.object({
    first_name: z.string().min(2).max(50),
    last_name: z.string().min(2).max(50),
  });

  const passwordFormSchema = z
    .object({
      password: z.string().min(8),
      confirm_password: z.string().min(8),
    })
    .refine((data) => data.password === data.confirm_password, {
      message: "Passwords don't match",
      path: ["confirm_password"],
    });

  const accountForm = useForm<z.infer<typeof accountFormSchema>>({
    resolver: zodResolver(accountFormSchema),
    defaultValues: { first_name: "", last_name: "" },
  });

  const passwordForm = useForm<z.infer<typeof passwordFormSchema>>({
    resolver: zodResolver(passwordFormSchema),
    defaultValues: { password: "", confirm_password: "" },
  });

  React.useEffect(() => {
    const getAccount = async () => {
      setState("loading");
      const { data, error } = await supabase
        .from("Account")
        .select()
        .eq("user_id", user?.id);

      // TODO: figure out what to do for non-email provider
      // supabase.auth.getUserIdentities().then(({ data, error }) => {
      //   for (const identity of data) {
      //     if (identity.provider === "google") {

      //     }
      //   }
      // });
      if (data?.length === 0) {
        setState("done");
        return;
      }
      if (error) {
        setState("error");
        toast.error("Failed getting your profile: " + (error as Error).message);
        return;
      }
      const { first_name, last_name } = data[0];
      accountForm.reset({
        first_name,
        last_name,
      });
      setState("done");
    };
    getAccount();

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onPasswordSubmit = React.useCallback(
    async (values: z.infer<typeof passwordFormSchema>, e?: React.FormEvent) => {
      e?.preventDefault();
      const result = passwordFormSchema.safeParse(values);
      if (result.success) {
        const { error } = await supabase.auth.updateUser({
          password: values.password,
        });
        if (error) {
          toast.error("Failed updating your password", {
            description: error.message,
          });
          return;
        }
        toast.success("Password updated!");
        await signOut();
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  const onAccountSubmit = React.useCallback(
    async (values: z.infer<typeof accountFormSchema>, e?: React.FormEvent) => {
      e?.preventDefault();
      const result = accountFormSchema.safeParse(values);
      if (result.success) {
        const prefillData = new Promise((resolve, reject) => {
          supabase
            .from("Account")
            .upsert({ ...values, user_id: user?.id })
            .select()
            .single()
            .then(({ data, error }) => {
              return error ? reject(error) : resolve({ data, error });
            });
        });
        toast.promise(prefillData, {
          loading: "Saving...",
          success: () => "Saved!",
          error: (error) => error.message,
        });
      } else {
        for (const issue of result.error.issues) {
          toast.error(issue.message);
        }
      }
    },

    [accountFormSchema, supabase, user?.id]
  );

  return (
    <div className="h-screen text-left">
      <h1 className="text-3xl text-center">Settings</h1>
      <Separator className="my-4" />
      {state === "done" ? (
        <>
          <Tabs
            value={tab}
            onValueChange={(_tab) => {
              setTab(_tab);
              navigate(`/settings/${_tab}`, { replace: true });
            }}
            className="w-[400px]"
          >
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="account">Account</TabsTrigger>
              <TabsTrigger value="password">Password</TabsTrigger>
              <TabsTrigger value="preferences">Preferences</TabsTrigger>
            </TabsList>
            <TabsContent value="account">
              <Card>
                <CardHeader>
                  <CardTitle>Account</CardTitle>
                  <CardDescription>
                    Make changes to your account here. Click save when you're
                    done.
                  </CardDescription>
                </CardHeader>
                <Form {...accountForm}>
                  <form
                    onSubmit={accountForm.handleSubmit((values) =>
                      onAccountSubmit(values)
                    )}
                  >
                    <CardContent className="space-y-2">
                      <div className="space-y-1">
                        <FormField
                          control={accountForm.control}
                          name="first_name"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel>First Name</FormLabel>
                              <FormControl>
                                <Input placeholder="John" {...field} />
                              </FormControl>
                              <FormDescription></FormDescription>
                              <FormMessage />
                            </FormItem>
                          )}
                        />
                      </div>
                      <div className="space-y-1">
                        <FormField
                          control={accountForm.control}
                          name="last_name"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel>Last Name</FormLabel>
                              <FormControl>
                                <Input placeholder="Doe" {...field} />
                              </FormControl>
                              <FormDescription></FormDescription>
                              <FormMessage />
                            </FormItem>
                          )}
                        />
                      </div>
                      <div className="grid w-full max-w-sm items-center gap-1.5 space-y-1">
                        <Label htmlFor="picture">Avatar</Label>
                        <Input id="picture" type="file" accept="image/*" />
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button
                        type="submit"
                        className="mt-4 w-full flex justify-center"
                      >
                        <SaveIcon />
                        <>Save Changes</>
                      </Button>
                    </CardFooter>
                  </form>
                </Form>
              </Card>
            </TabsContent>
            <TabsContent value="password">
              <Card>
                <CardHeader>
                  <CardTitle>Password</CardTitle>
                  <CardDescription>
                    Change your password here. After saving, you'll be logged
                    out.
                  </CardDescription>
                </CardHeader>
                <Form {...passwordForm}>
                  <form
                    onSubmit={passwordForm.handleSubmit((values) =>
                      onPasswordSubmit(values)
                    )}
                  >
                    <CardContent className="space-y-2">
                      <div className="space-y-1">
                        <FormField
                          control={passwordForm.control}
                          name="password"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel>New password</FormLabel>
                              <FormControl>
                                <Input type="password" {...field} />
                              </FormControl>
                              <FormDescription></FormDescription>
                              <FormMessage />
                            </FormItem>
                          )}
                        />
                      </div>
                      <div className="space-y-1">
                        <FormField
                          control={passwordForm.control}
                          name="confirm_password"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel>Confirm password</FormLabel>
                              <FormControl>
                                <Input type="password" {...field} />
                              </FormControl>
                              <FormDescription></FormDescription>
                              <FormMessage />
                            </FormItem>
                          )}
                        />
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button
                        type="submit"
                        className="mt-4 w-full flex justify-center"
                      >
                        <LockIcon />
                        Save password
                      </Button>
                    </CardFooter>
                  </form>
                </Form>
              </Card>
            </TabsContent>
            <TabsContent value="preferences">
              <Card>
                <CardHeader>
                  <CardTitle>Preferences</CardTitle>
                  <CardDescription>
                    Change your system preferences here.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <p className="text-left">Theme</p>
                      <div>
                        <Select
                          onValueChange={setTheme}
                          defaultValue={localStorage.theme}
                        >
                          <SelectTrigger className="w-[180px]">
                            <SelectValue placeholder="Select a theme" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectGroup>
                              <SelectLabel>Theme</SelectLabel>
                              <SelectItem value="system">System</SelectItem>
                              <SelectItem value="dark">Dark</SelectItem>
                              <SelectItem value="light">Light</SelectItem>
                            </SelectGroup>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      ) : (
        <div className="flex flex-col space-y-3">
          <Skeleton className="h-[350px] rounded-xl" />
          <div className="space-y-2">
            <Skeleton className="h-4 w-[250px]" />
            <Skeleton className="h-4 w-[200px]" />
          </div>
        </div>
      )}
    </div>
  );
}
