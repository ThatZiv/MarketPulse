import * as React from "react";
import {
  BookOpen,
  Command,
  LifeBuoy,
  Send,
  Settings2,
  ChartCandlestick,
  Gauge,
  CirclePlus,  
} from "lucide-react";

import { NavMain } from "@/components/nav-main";
import { NavSecondary } from "@/components/nav-secondary";
import { NavUser } from "@/components/nav-user";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { useSupabase } from "@/database/SupabaseProvider";
import { Skeleton } from "./ui/skeleton";
import { Button } from "./ui/button";
import { Link } from "react-router";

const data = {
  navMain: [
    {
      title: "Dashboard",
      url: "/",
      icon: Gauge,
      isActive: true,
    },
    {
      title: "Stocks",
      url: "/stocks",
      icon: CirclePlus,
      isActive: true,
    },
    {
      title: "Your Stocks",
      url: "/",
      icon: ChartCandlestick,
      items: [
        {
          title: "Ford",
          url: "/stocks/F",
        },
        {
          title: "GM",
          url: "/stocks/GM",
        },
        {
          title: "Tesla",
          url: "/stocks/TSLA",
        },
        {
          title: "Toyota",
          url: "/stocks/TM",
        },
        {
          title: "Rivian",
          url: "/stocks/RIVN",
        },
      ],
    },
    {
      title: "Documentation",
      url: "#",
      icon: BookOpen,
      items: [
        {
          title: "Introduction",
          url: "#",
        },
        {
          title: "Get Started",
          url: "#",
        },
        {
          title: "Tutorials",
          url: "#",
        },
        {
          title: "Changelog",
          url: "#",
        },
      ],
    },
    {
      title: "Settings",
      url: "/settings",
      icon: Settings2,
      items: [
        {
          title: "Account",
          url: "/settings/account",
        },
        {
          title: "Password",
          url: "/settings/password",
        },
        {
          title: "Preferences",
          url: "/settings/preferences",
        },
      ],
    },
  ],
  navSecondary: [
    {
      title: "Support",
      url: "#",
      icon: LifeBuoy,
    },
    {
      title: "Feedback",
      url: "#",
      icon: Send,
    },
  ],
};

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const { user, status } = useSupabase();
  return (
    <Sidebar variant="inset" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <a href="#">
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                  <Command className="size-4" />
                </div>
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold">MarketPulse</span>
                  {/* <span className="truncate text-xs"></span> */}
                </div>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.navMain} />
        {/* <NavProjects projects={data.projects} /> */}
        <NavSecondary items={data.navSecondary} className="mt-auto" />
      </SidebarContent>
      <SidebarFooter>
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
      </SidebarFooter>
    </Sidebar>
  );
}
