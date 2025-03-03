import * as React from "react";
import { actions } from "./constants";
import assert from "assert";

interface GlobalState {
  user: {
    id: string;
    email: string;
    name: string;
  };
}

const initialState: GlobalState = {
  user: {
    id: "",
    email: "",
    name: "",
  },
};

const GlobalReducer = (
  state: GlobalState,
  action: { type: number; payload: unknown }
): GlobalState => {
  switch (action.type) {
    case actions.SET_USER:
      if (typeof action.payload !== "object") {
        throw new Error("Expected object, got " + typeof action.payload);
      }
      // @ts-expect-error action.payload is an object
      return { ...state, user: action.payload };
    case actions.SET_USER_FULL_NAME:
      if (typeof action.payload !== "string") {
        throw new Error("Expected string, got " + typeof action.payload);
      }
      return { ...state, user: { ...state.user, name: action.payload } };
    default:
      throw new Error("Unknown action type: " + action.type);
  }
};

// eslint-disable-next-line react-refresh/only-export-components
export const GlobalContext = React.createContext<{
  state: GlobalState;
  dispatch: React.Dispatch<{ type: number; payload: unknown }>;
}>({ state: initialState, dispatch: () => {} });

export function GlobalProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = React.useReducer(GlobalReducer, initialState);
  console.log(state);
  return (
    <GlobalContext.Provider value={{ state, dispatch }}>
      {children}
    </GlobalContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export const useGlobal = () => React.useContext(GlobalContext);
