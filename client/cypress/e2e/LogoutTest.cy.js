describe("Logout Test", () => {
  it("Login and Logout of site", () => {
    cy.visit("http://localhost:5173/");
    const password = Cypress.env("password");
    const email = Cypress.env("email");

    cy.get('input[name="email"]').type(email);

    cy.get('input[name="password"]').type(password);

    cy.get("button").contains("Login").click();

    cy.get("button").contains("I Agree").click();

    cy.get('span[class="truncate text-xs"]').click();

    cy.get("div").contains("Log out").click();

    cy.get("div").contains("Log in");
    cy.get("div").contains("Enter your info below to login to your account");
  });
});
