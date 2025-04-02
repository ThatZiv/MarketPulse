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
import { useMemo, useState, useEffect } from "react";
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

  // Disclaimer addition
  const [showDisclaimer, setShowDisclaimer] = useState(false);

  useEffect(() => {
    if (user) {
      const hasAgreed = localStorage.getItem(
        `hasAgreedToDisclaimer_${user.id}`
      );
      setShowDisclaimer(hasAgreed !== "true");
    }
  }, [user]);

  const handleAgreeDisclaimer = () => {
    if (user) {
      localStorage.setItem(`hasAgreedToDisclaimer_${user.id}`, "true");
      setShowDisclaimer(false);
    }
  };

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
          className={`flex pt-4 justify-center transition-all duration-300 w-full bg-light-themed dark:bg-dark-themed bg-center bg-no-repeat bg-cover`}
        >
          <Outlet />
        </div>

        {/* Disclaimer addition*/}
        {showDisclaimer && user && location.pathname === "/" && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-lg w-full">
              <h2 className="text-xl font-bold mb-4">Disclaimer</h2>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-4">
                The stock information provided on this platform is for
                informational purposes only. By using this website, you
                acknowledge that any decisions you make based on the information
                provided are at your own risk. We are not responsible for any
                financial losses, damages, or liabilities incurred as a result
                of reliance on our content.
              </p>
              <div className="flex justify-end">
                <Button
                  onClick={handleAgreeDisclaimer}
                  className="bg-primary text-white"
                >
                  I Agree
                </Button>
              </div>
            </div>
          </div>
        )}
      </SidebarInset>
    </SidebarProvider>
  );
}
