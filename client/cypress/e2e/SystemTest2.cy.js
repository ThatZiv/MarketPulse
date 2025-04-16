describe("Stock Purchase History", () => {
  //Set to be skiped when run so it will not be run by git actions
  //Needs the python server running
  //Need to change client .env varriable VITE_API_URL from localhost to 0.0.0.0 so that it will connect to the python server
  //Change the bellow it.skip(...) to it(...)
  it.skip("Make Sure the purchase calculator is accurate based on users stock transaction history", () => {
    cy.visit("http://localhost:5173/");
    //Login
    const password = Cypress.env("password");
    const email = Cypress.env("email");

    cy.get('input[name="email"]').type(email);

    cy.get('input[name="password"]').type(password);

    cy.get("button").contains("Login").click();
    //Agree to disclaimer
    cy.get("button").contains("I Agree").click();

    //Click on TSLA stock
    cy.get("span").contains("TSLA").click();

    //
    cy.get("button").contains("Visit").click();

    // Make sure the server has time to load the graphs
    cy.wait(5000);

    cy.get("div").contains("TSLA Forecasts");
    cy.get("tspan").contains("Stock Price ($)");

    cy.get("div").contains("transformer");
    cy.get("div").contains("attention_lstm");
    cy.get("div").contains("cnn-lstm");
    cy.get("div").contains("XGBoost-model");
    cy.get("div").contains("az-sarima");
    cy.get("div").contains("average");
  });
});
