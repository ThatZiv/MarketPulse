/*eslint-disable @typescript-eslint/no-require-imports*/
const { defineConfig } = require("cypress");

/* eslint-disable @typescript-eslint/no-unused-vars */
module.exports = defineConfig({
  e2e: {
    setupNodeEvents(on, config) {
      // implement node event listeners here
    },
  },
});
