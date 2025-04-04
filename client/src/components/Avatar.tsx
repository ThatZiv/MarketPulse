import {
  Avatar as _Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/ui/avatar";
import { useSupabase } from "@/database/SupabaseProvider";
import { generateGravatarUrl } from "@/lib/utils";
import { Skeleton } from "./ui/skeleton";
import { useState, useEffect, useMemo } from "react";
import { useGlobal } from "@/lib/GlobalProvider";
import { actions } from "@/lib/constants";
export default function Avatar() {
  const { supabase, user, status } = useSupabase();
  const [imageReady, setImageReady] = useState(false);
  const { state, dispatch } = useGlobal();
  const [imagestatus, setImageStatus] = useState(status);
  const imageUrl = useMemo(() => {
    if (state.user.url == "") {
      return "";
    } else {
      setImageReady(true);
      setImageStatus("success");
      return state.user.url;
    }
  }, [state.user.url]);

  useEffect(() => {
    if (state.user.url === "") {
      setImageStatus("loading");
      const image_url = async () => {
        const { data } = await supabase
          .from("Account")
          .select("profile_picture")
          .eq("user_id", user?.id);
        if (data && data[0] && data[0].profile_picture) {
          const image = await supabase.storage
            .from("profile_pictures")
            .createSignedUrl(data[0].profile_picture, 3600);
          if (image.data) {
            setImageReady(true);
            state.user.url = image.data.signedUrl;
            dispatch({
              type: actions.SET_USER,
              payload: state.user,
            });
            setImageStatus("success");
          } else {
            setImageStatus("success");
          }
        } else {
          setImageStatus("success");
        }
      };
      image_url();
    } else {
      setImageReady(true);
      setImageStatus("success");
    }
  }, []);

  if (status === "loading" && imagestatus === "loading")
    return (
      <div>
        <Skeleton className="h-8 w-8 rounded-lg" />
      </div>
    );
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
