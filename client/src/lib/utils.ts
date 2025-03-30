import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { MD5 } from "crypto-js";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function generateGravatarUrl(id: string, size = 200) {
  const hash = MD5(id).toString();
  // https://www.gravatar.com/avatar/EMAIL_MD5?d=https%3A%2F%2Fui-avatars.com%2Fapi%2F/Lasse+Rafn/128 fallback
  return `https://www.gravatar.com/avatar/${hash}?s=${size}&d=identicon`;
}

export function capitalizeFirstLetter(str: string) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export function isSameDay(d1: Date, d2: Date) {
  return (
    d1.getDate() === d2.getDate() &&
    d1.getMonth() === d2.getMonth() &&
    d1.getFullYear() === d2.getFullYear()
  );
}
