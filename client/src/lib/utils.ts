import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import MD5 from "crypto-js/md5";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function generateGravatarUrl(id: string, size = 200) {
  const hash = MD5(id).toString();
  // https://www.gravatar.com/avatar/EMAIL_MD5?d=https%3A%2F%2Fui-avatars.com%2Fapi%2F/Lasse+Rafn/128 fallback
  return `https://www.gravatar.com/avatar/${hash}?s=${size}&d=identicon`;
}
