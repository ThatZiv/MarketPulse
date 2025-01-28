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
import { Button } from "@/components/ui/button";
export default function Dashboard() {
  const location = useLocation();
  const paths = useMemo(
    () => location.pathname.split("/"),
    [location.pathname]
  );
  return (
    <div>
      <SidebarProvider>
        <AppSidebar />
        <SidebarInset>
          <header className="flex h-16 shrink-0 items-center gap-2">
            <div className="flex items-center gap-2 px-4">
              <SidebarTrigger className="-ml-1" />
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
                            <BreadcrumbPage>{path}</BreadcrumbPage>
                          </BreadcrumbItem>
                        );
                      }
                      return (
                        <BreadcrumbItem key={index}>
                          <BreadcrumbLink
                            href={paths.slice(0, index + 1).join("/")}
                          >
                            {path}
                          </BreadcrumbLink>
                          <>
                            <BreadcrumbSeparator className="hidden md:block" />
                          </>
                        </BreadcrumbItem>
                      );
                    })
                    .filter(Boolean)}
                </BreadcrumbList>
              </Breadcrumb>
            </div>
          </header>
          <div className="flex items-center justify-center h-full">
            <Outlet />
          </div>
        </SidebarInset>
      </SidebarProvider>
      <Button
        onClick={async () => {
          const authToken = localStorage.getItem(
            "sb-xskipmrkpwewdbttinhd-auth-token"
          );
          if (authToken) {
            const token = JSON.parse(authToken);
            await fetch("http://127.0.0.1:5000/test", {
              method: "post",
              headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${token.access_token}`,
              },
            })
              .then((response) => response.json())
              .then((data) => console.log(data))
              .catch((error) => console.error("Error:", error));
          } else {
            console.log("Failed to find");
          }
        }}
        type="button"
      >
        Fetch Test
      </Button>
    </div>
  );
}
