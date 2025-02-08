import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Separator } from "@/components/ui/separator";
import { Link, Outlet, useLocation } from "react-router";
import { useMemo } from "react";
import React from "react";
import { useSupabase } from "@/database/SupabaseProvider";
import { NavUser } from "@/components/nav-user";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

const capitalize = (str: string) => {
  return str.charAt(0).toUpperCase() + str.slice(1);
};

export default function Dashboard() {
  const location = useLocation();
  const { user, status } = useSupabase();
  const paths = useMemo(
    () => location.pathname.split("/"),
    [location.pathname]
  );
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2">
          <div className="flex items-center gap-2 px-4">
            <SidebarTrigger className="-ml-1" />
            <Separator orientation="vertical" className="mr-2 h-4" />
            <Breadcrumb>
              <BreadcrumbList>
                <BreadcrumbItem>
                  <Link to="/" replace>
                    Home
                  </Link>
                </BreadcrumbItem>

                {paths.filter(Boolean).length > 0 && (
                  <BreadcrumbSeparator className="hidden md:block" />
                )}
                {paths
                  .map((path, index) => {
                    if (path === "") {
                      return null;
                    }
                    if (index === paths.length - 1) {
                      return (
                        <BreadcrumbItem key={index}>
                          <BreadcrumbPage>{capitalize(path)}</BreadcrumbPage>
                        </BreadcrumbItem>
                      );
                    }
                    return (
                      <React.Fragment key={index}>
                        <BreadcrumbItem>
                          <Link
                            className="transition-colors hover:text-foreground"
                            to={paths.slice(0, index + 1).join("/")}
                            replace
                          >
                            {capitalize(path)}
                          </Link>
                        </BreadcrumbItem>
                        <BreadcrumbSeparator className="hidden md:block" />
                      </React.Fragment>
                    );
                  })
                  .filter(Boolean)}
              </BreadcrumbList>
            </Breadcrumb>
          </div>
          <div className="flex justify-right gap-2 ml-auto">
            <div>
              {status === "loading" && (
                <Skeleton className="flex items-center justify-center h-16">
                  <div className="flex items-center space-x-4">
                    <Skeleton className="rounded-full h-10 w-10" />
                    <div className="flex flex-col">
                      <Skeleton className="w-24 h-4" />
                      <Skeleton className="w-16 h-3" />
                    </div>
                  </div>
                </Skeleton>
              )}
              {user ? (
                <NavUser />
              ) : (
                <Link to="/auth">
                  <Button className="w-full" size="lg">
                    Login
                  </Button>
                </Link>
              )}
            </div>
          </div>
        </header>
        <div
          className={`flex p-4 justify-center transition-all duration-300 w-full bg-light-themed dark:bg-dark-themed bg-center bg-no-repeat bg-cover`}
        >
          <Outlet />
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
