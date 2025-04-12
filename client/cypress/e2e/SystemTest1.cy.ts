describe("Stock Purchase History", () => {
  it("Make Sure the purchase calculator is accurate based on users stock transaction history", () => {
    cy.visit("http://localhost:5173/");

    cy.get('input[name="email"]').type("test2025@test.com");
    cy.get('input[name="password"]').type("Password123!");

    cy.get("button").contains("Login").click();

    cy.get("button").contains("I Agree").click();

    cy.get("span").contains("TSLA").click();

    cy.get("button").contains("Visit").click();

    cy.get("div").contains("Your TSLA Purchase History");

    // Chart elements
    cy.get("text").contains("Shares");

    // Other elememnts
    cy.get("div").contains("was lost from your last sale.");
    cy.get("div").contains("average price per share.");
    cy.get("div").contains("has been bought in total.");
    cy.get("div").contains("shares currently owned.");
  });
});
