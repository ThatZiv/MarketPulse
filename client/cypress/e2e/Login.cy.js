describe("Logim Test", () => {
  it("Login to the site", () => {
    cy.visit("http://localhost:5173/");
    const password = Cypress.env("password");
    const email = Cypress.env("email");

    cy.get('input[name="email"]').type(email);

    cy.get('input[name="password"]').type(password);

    cy.get("button").contains("Login").click();
  });
});
