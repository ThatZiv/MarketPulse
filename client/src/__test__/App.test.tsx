// src/__ tests __/App.test.tsx

import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import Login from "@/pages/Login";
import Create from "@/pages/Create";
import { BrowserRouter } from "react-router-dom";

test("demo", () => {
  expect(true).toBe(true);
});
// Needed to wrap in Router since it gets router App.tsx
test("Renders the login page", async () => {
  render(
    <BrowserRouter>
      <Login />
    </BrowserRouter>
  );

  const email = await screen.findByText("Email");
  expect(email).toBeInTheDocument();
});

test("Renders the create user page", async () => {
  render(
    <BrowserRouter>
      <Create />
    </BrowserRouter>
  );
  const email = await screen.findByText("Email");
  expect(email).toBeInTheDocument();
});
