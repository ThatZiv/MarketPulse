describe("Stock Estimation page (FR 32 // STC 2)", () => {
  //Set to be skiped when run so it will not be run by git actions
  //Needs the python server running
  //Need to change client .env varriable VITE_API_URL from localhost to 0.0.0.0 so that it will connect to the python server
  //Change the bellow it.skip(...) to it(...)
  it("Make sure the stock estimation page loads everything correctly (STC 2)", () => {
    cy.visit("http://localhost:5173/");
    //Login
    const password = Cypress.env("password");
    const email = Cypress.env("email");

    cy.get('input[name="email"]').type(email);

    cy.get('input[name="password"]').type(password);

    cy.get("button").contains("Login").click();

    cy.wait(2000);
    //Agree to disclaimer
    cy.get("button").contains("I Agree").click();

    //Click on TSLA stock
    cy.get("span").contains("TSLA").click();

    //
    cy.get("button").contains("Visit").click();

    // Make sure the server has time to load the graphs
    cy.wait(5000);

    cy.get("div").contains("TSLA Historical Prices");
    cy.get("h3").contains("Shares Owned");
    cy.get("h3").contains("Current Price");
    cy.get("p").contains(0);
    cy.get("div > p").contains("$");

    // forecasts
    cy.get("div").contains("TSLA Forecasts");
    cy.get("tspan").contains("Stock Price ($)");
    cy.get("div").contains("transformer");
    cy.get("div").contains("attention_lstm");
    cy.get("div").contains("cnn-lstm");
    cy.get("div").contains("XGBoost-model");
    cy.get("div").contains("az-sarima");
    cy.get("div").contains("average");

    cy.should("be.visible");

    // historical chart
    cy.get("#advanced").should("be.visible");
    cy.get("#advanced").should("be.enabled");
    cy.get("#advanced").click();
    cy.get("#condensed").click();

    // check if accuracy and other metrics are visible
    cy.get(".h-5 > :nth-child(1) > .font-medium").should("be.visible");
    cy.get(".h-5 > :nth-child(3) > .font-medium").should("be.visible");
    cy.get(":nth-child(5) > .font-medium").should("be.visible");
    cy.get("#advanced").click();

    // check if accuracy and other metrics are not visible AFTER clicking
    cy.get(".h-5 > :nth-child(1) > .font-medium").should("not.exist");
    cy.get(".h-5 > :nth-child(3) > .font-medium").should("not.exist");
  });

  it("Make sure the stock estimation page has hype and sentiment meters (ITC 6)", () => {
    cy.visit("http://localhost:5173/");
    //Login
    const password = Cypress.env("password");
    const email = Cypress.env("email");

    cy.get('input[name="email"]').type(email);

    cy.get('input[name="password"]').type(password);

    cy.get("button").contains("Login").click();

    cy.wait(2000);
    //Agree to disclaimer
    cy.get("button").contains("I Agree").click();

    //Click on TSLA stock
    cy.get("span").contains("TSLA").click();

    //
    cy.get("button").contains("Visit").click();

    // Make sure the server has time to load the graphs
    cy.wait(2000);

    // check if hype and sentiment meters are visible

    cy.get(
      ".grid-cols-6 > :nth-child(1) > .bg-card > .space-y-0 > .grid > .text-sm"
    ).should("be.visible");
    cy.get(
      ".grid-cols-6 > :nth-child(1) > .bg-card > .space-y-0 > .grid > .tracking-tight"
    ).should("be.visible");
    cy.get(
      ".grid-cols-6 > :nth-child(1) > .bg-card > .space-y-0 > .grid > .tracking-tight"
    ).should("have.text", "Hype Meter");
    cy.get(
      ":nth-child(2) > .bg-card > .space-y-0 > .grid > .tracking-tight"
    ).should("have.text", "Impact Factor");
    cy.get(":nth-child(2) > .bg-card > .pt-0 > .gap-2").should("be.visible");
    cy.get(":nth-child(1) > .bg-card > .pt-0 > .gap-2").should("be.visible");
    cy.get(
      ':nth-child(1) > .bg-card > .md\\:flex-row > .semicircle-gauge > svg > [transform="translate(66.68, 28)"] > .doughnut > :nth-child(6) > path'
    ).should("be.visible");
    cy.get(
      ":nth-child(1) > .bg-card > .md\\:flex-row > .semicircle-gauge > svg"
    ).should("be.visible");
    cy.get(
      ':nth-child(1) > .bg-card > .md\\:flex-row > .semicircle-gauge > svg > [transform="translate(66.68, 28)"] > .doughnut > :nth-child(4) > path'
    ).should("be.visible");
    cy.get(
      ':nth-child(2) > .bg-card > .md\\:flex-row > .semicircle-gauge > svg > [transform="translate(66.68, 28)"] > .doughnut > :nth-child(4) > path'
    ).should("be.visible");
  });
});
