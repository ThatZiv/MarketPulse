import {
  Avatar as _Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/ui/avatar";
import { useSupabase } from "@/database/SupabaseProvider";
import { generateGravatarUrl } from "@/lib/utils";
import { Skeleton } from "./ui/skeleton";
import useAsync from "@/hooks/useAsync";

interface picture {
  profile_picture: string;
}
export default function Avatar() {
  const { supabase, user, status } = useSupabase();
  let picture_key = "";
  const {
    value: picture_id,
    error: error,
    loading: loading,
  } = useAsync<picture[]>(
    () =>
      new Promise((resolve, reject) =>
        supabase
          .from("Account")
          .select("profile_picture")
          .eq("user_id", user?.id)
          .then(({ data, error }) => {
            if (error) reject(error);
            resolve(data || []);
          })
      ),
    [supabase]
  );

  if (picture_id) picture_key = picture_id[0].profile_picture;

  const { value: image_url } = useAsync(
    () =>
      new Promise((resolve, reject) => {
        if (picture_key != "") {
          supabase.storage
            .from("profile_pictures")
            .createSignedUrl(picture_key, 3600)
            .then(({ data, error }) => {
              if (error) reject(error);
              return resolve(data || []);
            });
        }
      })
  );
  let image: any = "";
  if (image_url) {
    console.log(image_url);
    image = image_url;
  }

  if (status === "loading") return <Skeleton className="h-8 w-8 rounded-lg" />;
  return (
    <_Avatar className="h-8 w-8 rounded-lg">
      {image_url ? (
        <AvatarImage src={image.signedUrl} alt="avatar" />
      ) : user ? (
        <AvatarImage src={generateGravatarUrl(user?.id)} alt="avatar" />
      ) : (
        <AvatarFallback className="rounded-lg">
          {(user!.email ?? "").charAt(0)}
        </AvatarFallback>
      )}
    </_Avatar>
  );
}
[0];
