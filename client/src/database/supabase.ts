import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_KEY;

// This may crash without valid keys,
export const supabase = createClient(supabaseUrl!, supabaseAnonKey!);
console.log("supabase client instance created");
//From supabase docs removed the redirect added variables
export async function signUpNewUser(email: string, password: string) {
  const { data, error } = await supabase.auth.signUp({
    email: email,
    password: password,
  });
  return { data, error };
}

//From supabase docs added variables
export async function signInWithEmail(email: string, password: string) {
  const { data, error } = await supabase.auth.signInWithPassword({
    email: email,
    password: password,
  });
  return { data, error };
}

//From supabase docs
export async function signOut() {
  const { error } = await supabase.auth.signOut();
  return { error };
}
