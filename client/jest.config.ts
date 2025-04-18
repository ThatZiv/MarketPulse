export default {
  preset: "ts-jest",
  testEnvironment: "jest-environment-jsdom",
  transform: {
    "^.+\\.tsx?$": "ts-jest",
  },
  setupFilesAfterEnv: ["<rootDir>/jest.setup.ts"],
  moduleNameMapper: {
    "^@/database/(.*)$": "<rootDir>/src/__test__/__mocks__/$1",
    "\\.css$": "identity-obj-proxy",
    "^@/(.*)$": "<rootDir>/src/$1",
  },
  globals: {
    "ts-jest": {
      diagnostics: {
        warnOnly: true,
      },
    },
  },
  testPathIgnorePatterns: ["<rootDir>/src/__test__/AddStock.test.tsx"],
};
