import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"

interface CustomTabProps {
    tabs: string[];
    tabcontent: React.ReactNode[];
  }
export function CustomTabs({ tabs,tabcontent }: CustomTabProps) {
  return (
    <Tabs defaultValue="account" className="w-[400px]">
      <TabsList className="grid w-full grid-cols-2">

      {tabs.map((tabname, index) => (
  <TabsTrigger key={index} value={tabname}>{tabname}</TabsTrigger>
))}


      </TabsList>
      {tabcontent.map(tab =>(
        <TabsContent value="account">
        {tab}
        </TabsContent>
      ) )}
      
    </Tabs>
  )
}
