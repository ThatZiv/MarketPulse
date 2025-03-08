import {
  Avatar as _Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/ui/avatar";
import { useSupabase } from "@/database/SupabaseProvider";
import { generateGravatarUrl } from "@/lib/utils";
import { Skeleton } from "./ui/skeleton";
import { useState, useEffect } from "react";
import { useGlobal } from "@/lib/GlobalProvider";
import { actions } from "@/lib/constants";
export default function Avatar() {
  const { supabase, user, status } = useSupabase();
  const [imageReady, setImageReady] = useState(false);
  const [imageUrl, setImageUrl] = useState("");
  const { state, dispatch } = useGlobal();

  useEffect(() => {
    console.log(state.user.url);
    console.log(state.user.name);
    console.log(state.user);
    if (state.user.url === "") {
      const image_url = async () => {
        const { data, error } = await supabase
          .from("Account")
          .select("profile_picture")
          .eq("user_id", user?.id);
        if (data) {
          const image = await supabase.storage
            .from("profile_pictures")
            .createSignedUrl(data[0].profile_picture, 3600);
          if (image.data) {
            console.log(image.data.signedUrl);
            setImageUrl(image.data.signedUrl);
            setImageReady(true);
            state.user.url = image.data.signedUrl;
            dispatch({
              type: actions.SET_USER,
              payload: [state.user],
            });
          }
        }
      };
      image_url();
    } else {
      setImageUrl(state.user.url);
      setImageReady(true);
    }
  }, []);

  if (status === "loading") return <Skeleton className="h-8 w-8 rounded-lg" />;
  return (
    <_Avatar className="h-8 w-8 rounded-lg">
      {imageReady ? (
        <AvatarImage src={imageUrl} alt="avatar" />
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
