export {};
type googleResponse = {
  clientId: string;
  client_id: string;
  credential: string;
  select_by: string;
};

declare global {
  interface Window {
    handleSignInWithGoogle: (response: googleResponse) => void;
  }
}
