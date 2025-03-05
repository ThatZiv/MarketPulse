import * as React from "react";
import { actions } from "./constants";
import { type GlobalState } from "@/types/global_state";

const initialState: GlobalState = {
  user: {
    id: "",
    email: "",
    name: "",
  },
  stocks: {},
  predictions: {},
  views: {
    predictions: {
      timeWindow: 7,
      model: null,
    },
  },
};

// general purpose reducer for the entire state
const GlobalReducer = (
  state: GlobalState,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  action: { type: number; payload: any }
): GlobalState => {
  switch (action.type) {
    case actions.SET_USER:
      if (typeof action.payload !== "object") {
        throw new Error("Expected object, got " + typeof action.payload);
      }
      return { ...state, user: action.payload };
    case actions.SET_USER_FULL_NAME:
      if (typeof action.payload !== "string") {
        throw new Error("Expected string, got " + typeof action.payload);
      }
      return { ...state, user: { ...state.user, name: action.payload } };
    // TODO move stock stuff into its own reducer (this is awful right now, i know)
    case actions.SET_STOCK_PRICE:
      if (typeof action.payload.stock_ticker !== "string") {
        throw new Error("Expected string for payload.stock_ticker");
      }
      if (typeof action.payload.data !== "number") {
        throw new Error("Expected number for payload.data");
      }
      return {
        ...state,
        stocks: {
          ...state.stocks,
          [action.payload.stock_ticker]: {
            ...state.stocks[action.payload.stock_ticker],
            current_price: action.payload.data,
            timestamp: Date.now(),
          },
        },
      };
    case actions.SET_STOCK_HISTORY:
      if (typeof action.payload.stock_ticker !== "string") {
        throw new Error("Expected string for payload.stock_ticker");
      }

      if (!Array.isArray(action.payload.data)) {
        throw new Error("Expected array for payload.data");
      }

      return {
        ...state,
        stocks: {
          ...state.stocks,
          [action.payload.stock_ticker]: {
            // TODO: finish this
            ...state.stocks[action.payload.stock_ticker],
            stock_name: action.payload.stock_name,
            history: action.payload.data,
            timestamp: Date.now(),
          },
        },
      };
    case actions.SET_PREDICTION:
      if (typeof action.payload.stock_ticker !== "string") {
        throw new Error("Expected string for payload.stock_ticker");
      }
      if (!Array.isArray(action.payload.data)) {
        throw new Error("Expected array for payload.data");
      }
      return {
        ...state,
        predictions: {
          ...state.predictions,
          [action.payload.stock_ticker]: action.payload.data,
        },
      };
    case actions.SET_PREDICTION_VIEW_TIME:
      if (typeof action.payload.timeWindow !== "number") {
        throw new Error("Expected number for payload.timeWindow");
      }

      return {
        ...state,
        views: {
          ...state.views,
          predictions: {
            ...state.views.predictions,
            timeWindow: action.payload.timeWindow,
          },
        },
      };
    case actions.SET_PREDICTION_VIEW_MODEL:
      if (typeof action.payload.model !== "string") {
        throw new Error("Expected string for payload.model");
      }
      return {
        ...state,
        views: {
          ...state.views,
          predictions: {
            ...state.views.predictions,
            model: action.payload.model,
          },
        },
      };

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
  return (
    <GlobalContext.Provider value={{ state, dispatch }}>
      {children}
    </GlobalContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export const useGlobal = () => React.useContext(GlobalContext);
