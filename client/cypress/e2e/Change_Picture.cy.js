describe("Change Profile Picture test", () => {
  it("Update the users profile picture", () => {
    cy.visit("http://localhost:5173/");
    const password = Cypress.env("password");
    const email = Cypress.env("email");

    cy.get('input[name="email"]').type(email);

    cy.get('input[name="password"]').type(password);

    cy.get("button").contains("Login").click();

    cy.get("button").contains("I Agree").click();

    cy.get("span").contains("Settings").click();

    cy.get('input[name="image"]').selectFile(
      "cypress/fixtures/MarketPulse_Logo.png"
    );

    cy.get("button").contains("Save Changes").click();
    cy.wait(10);
    cy.get("button").contains("Confirm").click();
    cy.wait(50);
  });
});
