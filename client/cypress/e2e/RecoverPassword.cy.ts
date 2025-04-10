describe("recover password test", () => {
  it("Render the login page to test the recover password page", () => {
    cy.visit("http://localhost:5173/");
    cy.get("button").contains("Forgot Password?").should("exist");
    cy.get("button").contains("Forgot Password?").click();

    // Now on the forgot password page

    cy.get("div").contains("Recover Password").should("exist");
    cy.get('input[name="email"]').type("FakeEmail");
    cy.get('input[name="email"]').should("have.value", "FakeEmail");

    cy.get("button").contains("Recover").click();

    cy.get("p").contains("Invalid email");
  });
});
