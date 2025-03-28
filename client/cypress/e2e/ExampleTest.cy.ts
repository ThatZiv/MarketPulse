describe('system test 1', () => {
  it('rendering the components of the login page', () => {
    cy.visit('http://localhost:5173/')
    cy.get('div').should('contain', 'Login');
    cy.get('div').contains('Enter your info below to login to your account').should('exist');
    cy.get('div').contains("Don't have an account? Create Account").should('exist');
    cy.get('div').contains('Login').should('exist');
    cy.get('button').contains('Forgot Password?').should('exist');
    cy.get('button[type="submit"]').should('contain', 'Login');
    cy.get('div').find('input[placeholder="Email Address"]').should('exist');
    cy.get('div').find('input[placeholder="Password"]').should('exist');
  })
})