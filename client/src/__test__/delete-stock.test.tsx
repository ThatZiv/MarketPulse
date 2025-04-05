import { DeleteStock } from "../components/delete-stock";
import { describe, test, beforeEach } from "@jest/globals";
import {
    render,
    screen,
} from "@testing-library/react";
import "@testing-library/jest-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom"; // Import MemoryRouter
import userEvent from '@testing-library/user-event';
import { Toaster } from "sonner";

jest.mock("lucide-react", () => ({
    AlertCircle: () => <span>AlertCircle Icon</span>,
}));
jest.mock('@/database/SupabaseProvider', () => ({
    useSupabase: () => ({
        user: { id: 'test-user-id', email: 'test@example.com' },
        supabase: {
            from: jest.fn().mockReturnValue({
                delete: jest.fn().mockReturnValue({
                    match: jest.fn().mockResolvedValueOnce({ data: [], error: null })
                }),
            }),
        },
    }),
}));

const queryClient = new QueryClient();

describe("DeleteStock Component", () => {

    beforeEach(() => {
        render(
            <MemoryRouter>
                <QueryClientProvider client={queryClient}>
                    <DeleteStock
                        ticker="Tesla"
                        stock_id={1}
                    />
                    <Toaster />
                </QueryClientProvider>
            </MemoryRouter>
        );
    });
    test("renders delete stock button correctly", () => {
        const deleteButton = screen.getByRole("button", { name: /Delete/i });
        expect(deleteButton).toBeInTheDocument();
    });
    test("renders alert dialog correctly", async () => {
        const deleteButton = screen.getByRole("button", { name: /Delete/i });
        expect(deleteButton).toBeInTheDocument();
        await userEvent.click(deleteButton);
        expect(await screen.findByText("Are you absolutely sure you want to delete the entire stock history?"))
            .toBeInTheDocument();
    });
    test("renders expected text on delete alert dialog box", async () => {
        const deleteButton = screen.getByRole("button", { name: /Delete/i });
        const headerText = "Are you absolutely sure you want to delete the entire stock history?";
        const warning = "Warning";
        const warningDescription = "This action cannot be undone. This will permanently delete your stock history and remove your data.";
        const alertTitle = "Deleting this stock results in:";
        const listItems = ["Your Stock purchase history will be permanently deleted.",
            "Stock predictions will be lost.",
            "All stored data for Tesla stock cannot be retrieved.",
            "To re-add Tesla stock, you'll need to enter the stock details again."];

        expect(deleteButton).toBeInTheDocument();
        await userEvent.click(deleteButton);
        const instruction = await screen.findByText(/Please enter/i);
        const span = instruction.querySelector("span");

        expect(instruction).toBeInTheDocument();
        expect(instruction).toHaveTextContent("Please enter");
        expect(instruction).toHaveTextContent("to confirm the deletion is intentional.");
        expect(span).toBeInTheDocument();
        expect(span).toHaveTextContent("Tesla");
        expect(span).toHaveClass("font-medium");

        expect(await screen.findByText(headerText)).toBeInTheDocument();
        expect(await screen.findByText(warning)).toBeInTheDocument();
        expect(await screen.findByText(warningDescription)).toBeInTheDocument();
        expect(await screen.findByText(alertTitle)).toBeInTheDocument();
        // expect(await heading2).toHaveTextContent(confirmationHeader);
        for (let item of listItems) {
            expect(await screen.findByText(item)).toBeInTheDocument();
        }
    });
    test("testing error toast", async () => {
        const deleteButton = screen.getByRole("button", { name: /Delete/i });
        const invalidInput = "Tesla Corp";
        expect(deleteButton).toBeInTheDocument();
        await userEvent.click(deleteButton);
        const inputPlaceHolder = screen.getByPlaceholderText("Tesla");
        expect(inputPlaceHolder).toBeInTheDocument();

        await userEvent.type(inputPlaceHolder, invalidInput);
        const confirmDeleteButton = screen.getByRole("button", { name: /I understand the consequences of removing this stock./i });
        await userEvent.click(confirmDeleteButton);
        const toastMessage = await screen.findByText("Stock name does not match. Please enter the correct stock name to delete.");
        expect(toastMessage).toBeInTheDocument();
    });
    test("testing successful deletion", async () => {
        const deleteButton = screen.getByRole("button", { name: /Delete/i });
        const invalidInput = "Tesla";
        expect(deleteButton).toBeInTheDocument();
        await userEvent.click(deleteButton);
        const inputPlaceHolder = screen.getByPlaceholderText("Tesla");
        expect(inputPlaceHolder).toBeInTheDocument();

        await userEvent.type(inputPlaceHolder, invalidInput);
        const confirmDeleteButton = screen.getByRole("button", { name: /I understand the consequences of removing this stock./i });
        await userEvent.click(confirmDeleteButton);
        const toastMessage = await screen.findByText("Deleting stock...");
        expect(toastMessage).toBeInTheDocument();
    });
});

