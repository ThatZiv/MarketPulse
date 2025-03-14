import { ResetPasswordForm } from "@/components/reset_password";
import { describe } from "node:test";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { expect, jest, test } from "@jest/globals";
import "@testing-library/jest-dom";
import { BrowserRouter } from "react-router-dom";
import React from "react";

afterEach(() => {
  cleanup();
});
jest.mock("lucide-react", () => ({
  Moon: () => "MoonIcon",
  Sun: () => "SunIcon",
  Eye: () => "EyeIcon",
  EyeOff: () => "EyeOffIcon",
}));
describe("Forgot Password", () => {
  test("Forgot Password rendering", async () => {
    render(
      <BrowserRouter>
        <ResetPasswordForm />
      </BrowserRouter>
    );
  });
});
