// This test does the same think twice to set and reset the name.
// This is split into to tests because it is faster
// Since the toast notifications will block the save changes button otherwise
// This is not a big issue in prod sine this test is much faster than
// would be possible for a typical user.

describe("Change first and last name", () => {
  it("Name Change 1", () => {
    cy.visit("http://localhost:5173/");

    const password = Cypress.env("password");
    const email = Cypress.env("email");

    cy.get('input[name="email"]').type(email);

    cy.get('input[name="password"]').type(password);

    cy.get("button").contains("Login").click();

    cy.get("button").contains("I Agree").click();

    cy.get("span").contains("Settings").click();
    cy.wait(200);
    cy.get('input[name="first_name"]').type("{selectAll}John");
    cy.wait(200);
    cy.get('input[name="last_name"]').type("{selectAll}Doe");

    cy.wait(50);
    cy.get("button").contains("Save Changes").click();
    cy.get("button").contains("Confirm").click();

    cy.get("b").contains("John Doe");
    cy.get("span").contains("John Doe");
  });

  it("Name change 2", () => {
    cy.visit("http://localhost:5173/");

    const password = Cypress.env("password");
    const email = Cypress.env("email");

    cy.get('input[name="email"]').type(email);

    cy.get('input[name="password"]').type(password);

    cy.get("button").contains("Login").click();

    cy.get("button").contains("I Agree").click();

    cy.get("span").contains("Settings").click();
    cy.wait(200);
    cy.get('input[name="first_name"]').type("{selectAll}Bill");
    cy.wait(200);
    cy.get('input[name="last_name"]').type("{selectAll}Smith");

    cy.wait(50);
    cy.get("button").contains("Save Changes").click();
    cy.get("button").contains("Confirm").click();
    cy.get("b").contains("Bill Smith");
    cy.get("span").contains("Bill Smith");
  });
});
