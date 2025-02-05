import * as React from "react";
import { useState, useEffect, useMemo } from "react";
import {
  BookOpen,
  Command,
  LifeBuoy,
  Send,
  Settings2,
  Gauge,
  CirclePlus,
  LucideProps,
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
import useAsync from "@/hooks/useAsync";

const data = {
  navMain: [
    {
      title: "Dashboard",
      url: "/",
      icon: Gauge,
      isActive: true,
      items: [],
    },
    {
      title: "Add New Stock",
      url: "/stocks",
      icon: CirclePlus,
      isActive: false,
      items: [],
    },
    {
      title: "Documentation",
      url: "#",
      icon: BookOpen,
      isActive: false,
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
      isActive: false,
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
      isActive: undefined,
      items: [],
    },
    {
      title: "Feedback",
      url: "#",
      icon: Send,
      isActive: undefined,
      items: [],
    },
  ],
};
interface NavItem {
  title: string;
  url: string;
  icon: React.ForwardRefExoticComponent<LucideProps & React.RefAttributes<SVGSVGElement>>;
  isActive: boolean | undefined;
  items: { title: string; url: string }[] | [] | undefined; // Allow null or undefined for items
}

interface NavData {
  navMain: NavItem[];
  navSecondary: NavItem[]; // Adjust based on your structure for secondary items
}

interface StockResponse {
  Stocks: {
    stock_name: string;
    stock_ticker: string;
  };
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const { user, status, supabase } = useSupabase();
  const [navData, setNavData] = useState<NavData>(data);

  const { value: stocks, error: stocksError } = useAsync<StockResponse[]>(
    () =>
      new Promise((resolve, reject) => {
        supabase
          .from("User_Stocks")
          .select("Stocks (stock_name, stock_ticker)")
          .eq("user_id", user?.id)
          .order("created_at", { ascending: false })
          .limit(5)
          .then(({ data, error }) => {
            if (error) reject(error);
            // @ts-expect-error Stocks will never expand to an array
            resolve(data || []);
          });
      }),
    [user, supabase]
  );
  const all_stocks = stocks?.map((stock) => ({
    title: stock.Stocks.stock_name,
    url: `/stocks/${stock.Stocks.stock_ticker}`,
  }));
  const stableStocks = useMemo(() => stocks, [stocks]);
  useEffect(() => {
    const updateStocks = () => {
      setNavData((prevData) => ({
        ...prevData,
        navMain: prevData.navMain.map((navItem) =>
          navItem.title === "Dashboard"
            ? { ...navItem, items: [...(navItem.items ?? []), ...((all_stocks ?? []).filter((stock) => 
              !navItem.items?.some(((item) => 
                item.url === stock.url))))] }
            : navItem
        ),
      }));
    };
    updateStocks();
  }, [stableStocks]);
  if (stocksError) {
    return (
      <div className="flex flex-col justify-center items-center h-screen">
        <h1 className="text-3xl">Error</h1>
        <p className="text-primary">
          Unfortunately, we encountered an error fetching your stocks. Please
          refresh the page or try again later.
        </p>
      </div>
    );
  }
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
        <NavMain items={navData.navMain} />
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
