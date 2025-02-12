import {
  Avatar as _Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/ui/avatar";
import { useSupabase } from "@/database/SupabaseProvider";
import { generateGravatarUrl } from "@/lib/utils";
import { Skeleton } from "./ui/skeleton";

export default function Avatar() {
  const { user, status } = useSupabase();
  if (status === "loading") return <Skeleton className="h-8 w-8 rounded-lg" />;
  return (
    <_Avatar className="h-8 w-8 rounded-lg">
      {user ? (
        <AvatarImage src={generateGravatarUrl(user?.id)} alt="avatar" />
      ) : (
        <AvatarFallback className="rounded-lg">
          {(user!.email ?? "").charAt(0)}
        </AvatarFallback>
      )}
    </_Avatar>
  );
}
