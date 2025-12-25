// src/api/httpClient.ts
import axios from "axios";
import { API_BASE_URL } from "../config/env";

export const http = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
});
