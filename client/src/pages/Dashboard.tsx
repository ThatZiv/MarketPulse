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
import { useMemo, useState } from "react";
export default function Dashboard() {
  const location = useLocation();
  const paths = useMemo(
    () => location.pathname.split("/"),
    [location.pathname]
  );

  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2">
          <div className="flex items-center gap-2 px-4">
            <SidebarTrigger className="-ml-1" onClick={toggleSidebar}/>
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
        <div className={`flex p-4 justify-center transition-all duration-300 w-full h-full "
          }`}>
          <Outlet />
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
