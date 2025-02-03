import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Separator } from "@/components/ui/separator";
import { Outlet, useLocation } from "react-router";
import { useMemo } from "react";
import React from "react";

const capitalize = (str: string) => {
  return str.charAt(0).toUpperCase() + str.slice(1);
};

export default function Dashboard() {
  const location = useLocation();
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
            <SidebarTrigger className="-ml-1"/>
            <Separator orientation="vertical" className="mr-2 h-4" />
            <Breadcrumb>
              <BreadcrumbList>
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
                          <BreadcrumbLink
                            href={paths.slice(0, index + 1).join("/")}
                          >
                            {capitalize(path)}
                          </BreadcrumbLink>
                        </BreadcrumbItem>
                        <BreadcrumbSeparator className="hidden md:block" />
                      </React.Fragment>
                    );
                  })
                  .filter(Boolean)}
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </header>
        <div className={`flex p-4 justify-center transition-all duration-300 w-full h-full bg-light-themed dark:bg-dark-themed bg-center bg-no-repeat bg-cover`}>
          <Outlet />
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
