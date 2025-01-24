// src/__ tests __/App.test.tsx

import "@testing-library/jest-dom";
import { render } from "@testing-library/react";
import Login from "@/pages/Login";

test("demo", () => {
  expect(true).toBe(true);
});

test("Renders the login page", () => {
  const { getByText } = render(<Login />);
  expect(getByText("Email")).toBeInTheDocument();
});
