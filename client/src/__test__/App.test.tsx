// src/__ tests __/App.test.tsx

import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import UserAuth from "@/pages/UserAuth";
import { BrowserRouter } from "react-router-dom";

test("demo", () => {
  expect(true).toBe(true);
});
jest.mock("lucide-react", () => ({
  Moon: () => "MoonIcon",
  Sun: () => "SunIcon",
}));

// Needed to wrap in Router since it gets router App.tsx
test("Renders the login page", async () => {
  render(
    <BrowserRouter>
      <UserAuth />
    </BrowserRouter>
  );

  const email = await screen.findByText("Email");
  expect(email).toBeInTheDocument();
});

test("Renders the create user page", async () => {
  render(
    <BrowserRouter>
      <UserAuth />
    </BrowserRouter>
  );
  const email = await screen.findByText("Email");
  expect(email).toBeInTheDocument();
});
