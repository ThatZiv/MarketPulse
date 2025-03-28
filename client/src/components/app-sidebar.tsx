import * as React from "react";
import { useState, useEffect } from "react";
import {
  BookOpen,
  LifeBuoy,
  Settings2,
  Gauge,
  CirclePlus,
  LucideProps,
} from "lucide-react";
import { NavMain } from "@/components/nav-main";
import { NavSecondary } from "@/components/nav-secondary";
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { useSupabase } from "@/database/SupabaseProvider";
import { Separator } from "./ui/separator";
import { Link } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { cache_keys } from "@/lib/constants";
import { Skeleton } from "./ui/skeleton";

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
      url: "/documentation",
      icon: BookOpen,
      isActive: false,
      items: [
        {
          title: "Introduction",
          url: "/documentation/introduction",
        },

        {
          title: "Tutorials",
          url: "/documentation/tutorials",
        },
        {
          title: "FAQ",
          url: "/documentation/faq",
        },
        {
          title: "Disclaimer",
          url: "/documentation/disclaimer",
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
      url: "",
      icon: LifeBuoy,
      isActive: undefined,
      items: [],
    },
    {
      title: "Feedback",
      url: "/feedback",
      icon: LifeBuoy,
      isActive: undefined,
      items: [],
    },
  ],
};
interface NavItem {
  title: string;
  url: string;
  icon: React.ForwardRefExoticComponent<
    LucideProps & React.RefAttributes<SVGSVGElement>
  >;
  isActive: boolean | undefined;
  items: { title: string; url: string }[] | [] | undefined;
}

interface NavData {
  navMain: NavItem[];
  navSecondary: NavItem[];
}

interface StockResponse {
  Stocks: {
    stock_id: number;
    stock_name: string;
    stock_ticker: string;
  };
  shares_owned: number;
}

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const [navData, setNavData] = useState<NavData>(data);
  const { supabase, user } = useSupabase();
  const {
    data: stocks,
    error: stocksError,
    status: stocksStatus,
  } = useQuery<StockResponse[]>({
    queryKey: [cache_keys.USER_STOCKS],
    queryFn: () =>
      new Promise((resolve, reject) => {
        supabase
          .from("User_Stocks")
          .select("Stocks (*), shares_owned")
          .eq("user_id", user?.id)
          .order("created_at", { ascending: false })
          .limit(5)
          .then(({ data, error }) => {
            if (error) reject(error);
            // @ts-expect-error Stocks will never expand to an array
            resolve(data || []);
          });
      }),
  });
  useEffect(() => {
    if (!stocks || stocks.length === 0) return;

    const formattedStocks = stocks.map((stock) => ({
      title: stock.Stocks.stock_name,
      url: `/stocks/${stock.Stocks.stock_ticker}`,
    }));

    setNavData((prevData) => ({
      ...prevData,
      navMain: prevData.navMain.map((navItem) => {
        if (navItem.title === "Dashboard") {
          return {
            ...navItem,
            items: [
              ...(navItem.items ?? []).filter(item =>
                formattedStocks.some(stock => stock.url === item.url)
              ),
              ...formattedStocks.filter(
                (stock) => !navItem.items?.some(item => item.url === stock.url)
              ),
            ],
          };
        }
        return navItem;
      }),
    }));

  }, [stocks]);

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

  const loading = stocksStatus === "pending";

  return (
    <Sidebar variant="inset" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <Link to={window.location.origin}>
                <div className="flex items-center">
                  <img
                    src="/public/images/MarketPulse_Logo.png"
                    alt="MarketPulse Logo"
                    className="h-10 w-10 mr-2"
                  />
                  <div className="grid flex-1 text-left text-sm leading-tight">
                    <span className="truncate font-semibold">MarketPulse</span>
                  </div>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
        <Separator />
      </SidebarHeader>
      <SidebarContent>
        {loading ? (<>
          {new Array(4).fill(undefined).map((_, index) => (
            <Skeleton
              key={index}
              className="w-11/12 h-7 mx-auto bg-gray-200 dark:bg-gray-700"
            />
          ))}
        </>) :
          (<NavMain items={navData.navMain} />)}
        <NavSecondary items={data.navSecondary} className="mt-auto" />
      </SidebarContent>
    </Sidebar>
  );
}
