import axios, { type AxiosInstance, type AxiosError } from "axios";

export interface IApi {
  getStockLogo: (ticker: string) => Promise<string>;
  get: <TRes>(path: string) => Promise<TRes>;
  post: <TReq, TRes>(path: string, object?: TReq) => Promise<TRes>;
}

export default class Api implements IApi {
  private instance: AxiosInstance;
  private token?: string;

  constructor(token?: string, instance?: AxiosInstance) {
    this.token = token;
    this.instance = instance || this.createInstance();
  }

  protected createInstance() {
    return axios.create({
      baseURL: import.meta.env.VITE_API_URL,
      headers: this.token ? { Authorization: `Bearer ${this.token}` } : {},
    });
  }

  /**
   * make custom request to the backend
   * @param path path to get
   * @returns response
   */
  public async get<TRes>(path: string) {
    try {
      const resp = await this.instance.get(path);
      return resp.data as TRes;
    } catch (error) {
      handleServiceError(error as AxiosError);
    }
    return {} as TRes;
  }
  /**
   * post custom request to backend
   * @param path The path to post to
   * @param object The object to post
   * @returns response
   */
  public async post<TReq, TRes>(path: string, object?: TReq) {
    try {
      const resp = await this.instance.post<TRes>(path, object);
      return resp.data as TRes;
    } catch (error) {
      handleServiceError(error as AxiosError);
    }
    return {} as TRes;
  }
  /**
   * get the stock logo for a given ticker
   * @param ticker stock ticker
   * @example AAPL
   * @returns stock logo as a blob
   */

  public async getStockLogo(ticker: string) {
    // return this.instance.get(`/auth/logo?ticker=${ticker}`);
    try {
      const resp = await this.instance.get(`/auth/logo?ticker=${ticker}`, {
        responseType: "arraybuffer",
      });
      const img = new Blob([resp.data], { type: "image/png" });
      return URL.createObjectURL(img);
    } catch (error) {
      handleServiceError(error as AxiosError);
    }
    return "";
  }
}

function handleServiceError(error: AxiosError) {
  if (error.response) {
    const status = error.response.status;
    if (status === 401) {
      // unauthorized error
    } else if (status === 403) {
      // forbidden error
    } else if (status === 404) {
      // not found error
    } else if (status === 500) {
      // internal server error
    } else if (status === 503) {
      // service unavailable error
    } else if (status === 504) {
      // gateway timeout error
    } else {
      // other errors
    }
  }
}
