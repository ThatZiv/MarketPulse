import 'cypress';

declare global {
  namespace Cypress {
    interface Cypress {
      env: {
        email: string;
        password: string;
      };
    }
  }
}