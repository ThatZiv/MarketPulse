describe("Change Password test", () => {
  beforeEach(() => {
    // increase screen res so toast does not block buttons
    cy.viewport(1920, 1080);
  });
  // Changes the password then tests it by it changing back
  it("Change Users password and use it to login", () => {
    cy.visit("http://localhost:5173/");
    const password = Cypress.env("password");
    const email = Cypress.env("email");

    cy.get('input[name="email"]').type(email);

    cy.get('input[name="password"]').type(password);

    cy.get("button").contains("Login").click();

    cy.get("button").contains("I Agree").click();

    cy.get("span").contains("Settings").click();

    cy.get("button").contains("Password").click();

    cy.get('input[name="old_password"]').type(password);

    cy.get('input[name="password"]').type(password + "new");
    cy.get('input[name="confirm_password"]').type(password + "new");

    cy.get("button").contains("Save Password").click();
    cy.wait(10);
    cy.get("button").contains("Confirm").click();
    cy.wait(50);

    cy.get('input[name="email"]').type(email);

    cy.get('input[name="password"]').type(password + "new");

    cy.get("button").contains("Login").click();

    cy.get("span").contains("Settings").click();

    cy.get("button").contains("Password").click();

    cy.get('input[name="old_password"]').type(password + "new");

    cy.get('input[name="password"]').type(password);
    cy.get('input[name="confirm_password"]').type(password);

    cy.get("button").contains("Save Password").click();
    cy.wait(10);
    cy.get("button").contains("Confirm").click();
    // neeed to wait so that request gets through
    cy.wait(500)
  });
});
