import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useSupabase } from "@/database/SupabaseProvider";
import { LockIcon, SaveIcon, Eye, EyeOff } from "lucide-react";
import React, { useState } from "react";
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
import { useGlobal } from "@/lib/GlobalProvider";
import { actions } from "@/lib/constants";
//import { profile } from "node:console";
import { v4 as uuidv4 } from "uuid";

export default function SettingsPage() {
  const { session, supabase, user, signOut } = useSupabase();
  const { dispatch } = useGlobal();
  const { state: globalState } = useGlobal();
  const [state, setState] = React.useState<"loading" | "error" | "done">(
    "loading"
  );
  const { tab: tabParam } = useParams();
  const { setTheme } = useTheme();
  const navigate = useNavigate();
  const [tab, setTab] = React.useState(tabParam ?? "account");
  const [password, setPassword] = useState("");
  const [, setOldPassword] = useState("");
  const [, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showOldPassword, setShowOldPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [fontSize, setFontSize] = useState(
    localStorage.getItem("fontSize") || "medium"
  );

  const accountFormSchema = z.object({
    first_name: z.string().min(2).max(50),
    last_name: z.string().min(2).max(50),
    image: z.instanceof(FileList).optional(),
  });

  const passwordFormSchema = z
    .object({
      old_password: z.string().min(1, "Old password is required"),
      password: z
        .string()
        .min(8, "")
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
      confirm_password: z.string().min(8, ""),
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
    defaultValues: { old_password: "", password: "", confirm_password: "" },
  });

  React.useEffect(() => {
    const getAccount = async () => {
      setState("loading");
      const { data, error } = await supabase
        .from("Account")
        .select()
        .eq("user_id", user?.id);

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
  }, []);

  type AccountFormValues = z.infer<typeof accountFormSchema>;
  type PasswordFormValues = z.infer<typeof passwordFormSchema>;

  const onPasswordSubmit = async (values: PasswordFormValues) => {
    const { error } = await supabase.auth.signInWithPassword({
      email: user?.email ?? "",
      password: values.old_password,
    });

    if (error) {
      toast.error("Old password is incorrect", {
        description: error.message,
      });
      return;
    }

    toast("Are you sure you want to change your password?", {
      action: {
        label: "Confirm",
        onClick: async () => {
          const { error } = await supabase.auth.updateUser({
            password: values.password,
          });

          if (error) {
            toast.error("Failed updating your password", {
              description: error.message,
            });
          } else {
            toast.success("Password updated! Logging out...");
            await signOut();
          }
        },
      },
    });
  };

  const fileRef = accountForm.register("image");
  const onAccountSubmit = async (values: AccountFormValues) => {
    console.log(values.image);
    toast("Are you sure you want to confirm these changes?", {
      action: {
        label: "Confirm",
        onClick: async () => {
          if (values.image!.length > 0) {
            const file = uuidv4();
            const response = await supabase.storage
              .from("profile_pictures")
              .upload(file, values.image![0], {
                upsert: true,
                headers: {
                  Authorization: `Bearer ${session?.access_token}`,
                },
              });
            if (!response.error) {
              const response = await supabase.storage
                .from("profile_pictures")
                .createSignedUrl(file, 3600);
              const { error } = await supabase
                .from("Account")
                .upsert({
                  profile_picture: file,
                  first_name: values.first_name,
                  last_name: values.last_name,
                  user_id: user?.id,
                })
                .select()
                .single();

              if (error) {
                toast.error("Failed updating your profile", {
                  description: error.message,
                });
              } else if (response.error) {
                toast.error("Failed updating your profile", {
                  description: response.error.message,
                });
              } else {
                toast.success("Profile updated successfully!");
                globalState.user.url = response.data.signedUrl;
                globalState.user.name =
                  values.first_name + " " + values.last_name;
                dispatch({
                  type: actions.SET_USER,
                  payload: globalState.user,
                });

                navigate("/");
              }
            } else {
              toast.error("Failed updating your profile", {
                description: response.error.message,
              });
            }
          } else {
            const { error } = await supabase
              .from("Account")
              .upsert({
                first_name: values.first_name,
                last_name: values.last_name,
                user_id: user?.id,
              })
              .select()
              .single();

            if (error) {
              toast.error("Failed updating your profile", {
                description: error.message,
              });
            } else {
              toast.success("Profile updated successfully!");
              dispatch({
                type: actions.SET_USER_FULL_NAME,
                payload: [values.first_name, values.last_name]
                  .filter((x) => x)
                  .join(" ")
                  .trim(),
              });
              navigate("/");
            }
          }
        },
      },
    });
  };

  const passwordValidations = [
    { text: "At least 8 characters", isValid: password.length >= 8 },
    { text: "At least 1 uppercase letter", isValid: /[A-Z]/.test(password) },
    { text: "At least 1 number", isValid: /[0-9]/.test(password) },
    { text: "At least 1 special character", isValid: /[\W_]/.test(password) },
  ];

  const handleFontSizeChange = (value: string) => {
    setFontSize(value);
    localStorage.setItem("fontSize", value);
    document.documentElement.style.fontSize =
      value === "small" ? "14px" : value === "large" ? "18px" : "16px";
  };

  return (
    <div className="h-screen text-left">
      <h1 className="text-3xl text-center">Settings</h1>
      <Separator className="my-4" />
      {state === "done" ? (
        <>
          <Tabs
            value={tab}
            onValueChange={(_tab: string) => {
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
                      <div className="space-y-1">
                        <FormField
                          control={accountForm.control}
                          name="image"
                          rules={{ required: "File is required" }}
                          render={() => (
                            <FormItem>
                              <FormLabel>Avatar</FormLabel>
                              <FormControl>
                                <Input
                                  type="file"
                                  accept="image/*"
                                  {...fileRef}
                                />
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
                          name="old_password"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel>Old password</FormLabel>
                              <FormControl>
                                <div className="relative">
                                  <Input
                                    type={showOldPassword ? "text" : "password"}
                                    {...field}
                                    onChange={(e) => {
                                      field.onChange(e);
                                      setOldPassword(e.target.value);
                                    }}
                                  />
                                  <button
                                    type="button"
                                    onClick={() =>
                                      setShowOldPassword((prev) => !prev)
                                    }
                                    className="absolute inset-y-0 right-2 flex items-center text-gray-500"
                                  >
                                    {showOldPassword ? (
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
                      </div>
                      <div className="space-y-1">
                        <FormField
                          control={passwordForm.control}
                          name="password"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel>New password</FormLabel>
                              <FormControl>
                                <div className="relative">
                                  <Input
                                    type={showPassword ? "text" : "password"}
                                    {...field}
                                    onChange={(e) => {
                                      field.onChange(e);
                                      setPassword(e.target.value);
                                    }}
                                  />
                                  <button
                                    type="button"
                                    onClick={() =>
                                      setShowPassword((prev) => !prev)
                                    }
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
                                      req.isValid
                                        ? "text-green-500"
                                        : "text-red-500"
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
                      </div>
                      <div className="space-y-1">
                        <FormField
                          control={passwordForm.control}
                          name="confirm_password"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel>Confirm password</FormLabel>
                              <FormControl>
                                <div className="relative">
                                  <Input
                                    type={
                                      showConfirmPassword ? "text" : "password"
                                    }
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
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button
                        type="submit"
                        className="mt-4 w-full flex justify-center"
                      >
                        <LockIcon />
                        Save Password
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
                    <div className="flex items-center justify-between mt-4">
                      <p className="text-left">Font Size</p>
                      <div>
                        <Select
                          onValueChange={handleFontSizeChange}
                          defaultValue={fontSize}
                        >
                          <SelectTrigger className="w-[180px]">
                            <SelectValue placeholder="Select a font size" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectGroup>
                              <SelectLabel>Font Size</SelectLabel>
                              <SelectItem value="small">Small</SelectItem>
                              <SelectItem value="medium">Medium</SelectItem>
                              <SelectItem value="large">Large</SelectItem>
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
