import { ResetPassword } from "@/components/forgot-password";
import { describe } from "node:test";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { expect, jest, test } from "@jest/globals";
import "@testing-library/jest-dom";
import React from "react";

afterEach(() => {
  cleanup();
});

describe("Forgot Password", () => {
  const login = jest.fn();

  test("Forgot Password rendering", async () => {
    render(<ResetPassword resetPasswordState={login} />);
  });
});
