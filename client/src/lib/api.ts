import axios, { type AxiosInstance, type AxiosError } from "axios";
import { toast } from "sonner";

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
      this.handleError(error as AxiosError);
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
      this.handleError(error as AxiosError);
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
    try {
      const resp = await this.instance.get(`/auth/logo?ticker=${ticker}`, {
        responseType: "arraybuffer",
      });
      const img = new Blob([resp.data], { type: "image/png" });
      return URL.createObjectURL(img);
    } catch (error) {
      this.handleError(error as AxiosError);
    }
    return "";
  }

  /**
   * handle error from axios
   * @param error  axios error
   */
  protected async handleError(error: AxiosError) {
    const status = error.response?.status;
    // TODO: find a way to make these customizable (duration, dismiss, etc.)
    switch (status) {
      case 400:
        toast.error("Bad Request", {
          description: "The request was invalid or cannot be served",
        });
        break;
      case 401:
        toast.error("Unauthorized", {
          description: "You are not authorized to perform this action",
        });
        break;
      case 403:
        toast.error("Forbidden", {
          description: "You are not allowed to perform this action",
        });
        break;
      case 404:
        toast.error("Not Found", {
          description: "The requested resource was not found",
        });
        break;
      case 500:
        toast.error("Internal Server Error", {
          description: "An error occurred on the server",
        });
        break;
      case 503:
        toast.error("Service Unavailable", {
          description: "The service is currently unavailable",
        });
        break;
      case 504:
        toast.error("Timeout", {
          description: "The server took too long to respond",
        });
        break;
      default:
        toast.error("Error", {
          description: error.message,
        });
    }
  }
}
