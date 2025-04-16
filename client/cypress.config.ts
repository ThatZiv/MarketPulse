/*eslint-disable @typescript-eslint/no-require-imports*/
const { defineConfig } = require("cypress");
require("dotenv").config();

/* eslint-disable @typescript-eslint/no-unused-vars */
module.exports = defineConfig({
  e2e: {
    setupNodeEvents(on, config) {
      // implement node event listeners here
    },
    env: {
      email: process.env.TEST_LOGIN_EMAIL,
      password: process.env.TEST_LOGIN_PASSWORD,
    },
  },
});
