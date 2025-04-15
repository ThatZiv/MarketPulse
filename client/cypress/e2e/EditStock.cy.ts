describe("Edit Stock Test", () => {
  it("Edit Stock", () => {
    cy.visit("http://localhost:5173/");

    const password = Cypress.env("password");
    const email = Cypress.env("email");

    cy.get('input[name="email"]').type(email);

    cy.get('input[name="password"]').type(password);

    cy.get("button").contains("Login").click();

    cy.get("button").contains("I Agree").click();

    cy.get("span").contains("TSLA").click();

    cy.get("button").contains("Edit").click();

    cy.get("button").contains("Add Transaction").click();

    cy.get('input[id="date-2"]').type("2023-12-02T10:30");
    cy.get('input[id="shares-2"]').type("10");
    cy.get('input[id="price-2"]').type("12");
    cy.wait(100);
    cy.get("button").contains("Submit").click();
    cy.wait(100);

    // Delete added value to reset
    cy.get("span").contains("TSLA").click();
    cy.get("button").contains("Edit").click();

    cy.get(":nth-child(3) > :nth-child(2) > .inline-flex").click();
    cy.wait(1000);
    cy.get("button").contains("Submit").click();
  });
});
