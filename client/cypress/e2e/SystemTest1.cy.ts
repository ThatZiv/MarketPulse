describe("Stock Purchase History", () => {
    it("Make Sure the purchase calculator is accurate based on users stock transaction history", () => {
      cy.visit("http://localhost:5173/");

      cy.get('input[name="email"]').type("test2025@test.com");
      cy.get('input[name="password"]').type("Password123!");
      
      cy.get("button").contains("Login").click();

      cy.get("button").contains("I Agree").click();

      cy.get("span").contains("TSLA").click();

      cy.get("button").contains("Visit").click();

      cy.get("div").contains("Your TSLA Purchase History")

    // Table elements
        cy.get("td").contains("March 23, 2025 04:00 PM")
        cy.get("td").contains("Buy")
        cy.get("td").contains("4")
        cy.get("td").contains("$378.00")
        cy.get("td").contains("April 09, 2025 02:15 PM")
        cy.get("td").contains("Sell")
        cy.get("td").contains("3")
        cy.get("td").contains("$335.00")
        
    // Chart elements
        cy.get("text").contains("Shares")
        cy.get("text").contains("Mar 23 04:00 PM")
        cy.get("text").contains("Apr 9 02:15 PM")

    // Other elememnts
        cy.get("div").contains("was lost from your last sale.")
        cy.get("div").contains("average price per share.")
        cy.get("div").contains("has been bought in total.")
        cy.get("div").contains("shares currently owned.")
        cy.get("div").contains("$129.00")
        cy.get("div").contains("$378.00")
        cy.get("div").contains("$1,512.00")
        cy.get("div").contains("1")
    });
  });
  