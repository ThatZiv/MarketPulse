export const supabase = 0;

//From supabase docs removed the redirect added variables
export async function signUpNewUser(email: string, password: string) {
  const data = email;
  const error = password;
  return { data, error };
}

//From supabase docs added variables
export async function signInWithEmail(email: string, password: string) {
  const data = email;
  const error = password;
  return { data, error };
}

//From supabase docs
export async function signOut() {
  const error = 1;
  return { error };
}
