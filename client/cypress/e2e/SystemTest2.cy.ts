describe("Stock Purchase History", () => {
  //Set to be skiped when run so it will not be run by git actions
  //Needs the python server running to work
  //Need to change .env VITE_API_URL from localhost to 0.0.0.0 so that it will connect to the python server
  it.skip("Make Sure the purchase calculator is accurate based on users stock transaction history", () => {
    cy.visit("http://localhost:5173/");
    //Login
    cy.get('input[name="email"]').type("test2025@test.com");
    cy.get('input[name="password"]').type("Password123!");

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
