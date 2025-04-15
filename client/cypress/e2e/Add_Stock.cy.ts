describe("Dark mode test", () => {
  it("Change between dark mode and light mode logged out", () => {
    cy.visit("http://localhost:5173/");
    const password = Cypress.env("password");
    const email = Cypress.env("email");

    cy.get('input[name="email"]').type(email);

    cy.get('input[name="password"]').type(password);

    cy.get("button").contains("Login").click();

    cy.get("button").contains("I Agree").click();

    cy.get("a").contains("+").click();
    cy.get(".rounded-lg > :nth-child(1) > .flex").click();
    cy.get(":nth-child(5) > .relative").click();
    cy.get("#hasStocks").click();

    cy.get("#cashToInvest").type("1000");

    cy.get("button").contains("Add Transaction").click();

    cy.get('input[id="date-0"]').type("2023-12-02T10:30");
    cy.get('input[id="shares-0"]').type("10");
    cy.get('input[id="price-0"]').type("12");
    cy.wait(100);

    cy.get("button").contains("Submit").click();
    cy.wait(100);

    // Cleanup stock that was added
    cy.get("span").contains("TM").click();
    cy.get("button").contains("Visit").click();

    cy.get("button").contains(" Delete").click();

    cy.get('input[placeholder="Toyota Motor Corporation"]').type(
      "Toyota Motor Corporation"
    );

    cy.get("button")
      .contains("I understand the consequences of removing this stock.")
      .click();
  });
});
