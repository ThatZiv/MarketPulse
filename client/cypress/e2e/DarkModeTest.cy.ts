describe("Dark mode test", () => {
  it("Change between dark mode and light mode logged out", () => {
    cy.visit("http://localhost:5173/");

    cy.get('button[aria-haspopup="menu"]').click();
    cy.get("div").contains("Light").click();
    cy.get('html[class="light"]');
    cy.wait(200);

    cy.get('button[aria-haspopup="menu"]').click();
    cy.get("div").contains("Dark").click();
    cy.get('html[class="dark"]');
  });

  it("Change between dark mode and light mode logged out", () => {
    cy.visit("http://localhost:5173/");
    const password = Cypress.env("password");
    const email = Cypress.env("email");

    cy.get('input[name="email"]').type(email);

    cy.get('input[name="password"]').type(password);

    cy.get("button").contains("Login").click();

    cy.get("button").contains("I Agree").click();

    cy.get('span[class="truncate text-xs"]').click();
    cy.wait(150);
    cy.get('button[aria-haspopup="menu"][aria-expanded="false"]').click();
    cy.get("div").contains("Light").click();
    cy.get('html[class="light"]');

    cy.wait(150);
    cy.get('button[aria-haspopup="menu"][aria-expanded="false"]').click();
    cy.get("div").contains("Dark").click();
    cy.get('html[class="dark"]');
  });
});
