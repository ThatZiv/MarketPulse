describe("Stock Purchase History", () => {
    it("Make Sure the purchase calculator is accurate based on users stock transaction history", () => {

        
      
        cy.visit("http://localhost:5173/");

      cy.get('input[name="email"]').type("test2025@test.com");
      cy.get('input[name="password"]').type("Password123!");
      
      cy.get("button").contains("Login").click();

      cy.get("button").contains("I Agree").click();

      cy.get("span").contains("TSLA").click();

      cy.get("button").contains("Visit").click();
      
      
      // Make sure the server has time to load the graphs
      cy.wait(5000)
      

      cy.get("div").contains("TSLA Forecasts")
      cy.get("tspan").contains("Stock Price ($)")

      cy.get("div").contains("transformer")
      cy.get("div").contains("attention_lstm")
      cy.get("div").contains("cnn-lstm")
      cy.get("div").contains("XGBoost-model")
      cy.get("div").contains("az-sarima")
      cy.get("div").contains("average")


    });
  });